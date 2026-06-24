import hydra
from omegaconf import DictConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

from src.node.state_and_dataclass import SharedState
from src.node.data_gatherer import data_gatherer_node
from src.node.memory_retrieval import SummaryDatabase
from src.node.summary_generator import SummaryGenerator

# load environment variable
load_dotenv()


# conditonal edges
def check_gathered_data_present(state: SharedState) -> str:
    if not state["gathered_data"]:
        return "end"
    return "memory retrieval"


# function to build graph (to add other config later)
def build_graph(database_cfg: DictConfig, node_cfg: DictConfig) -> CompiledStateGraph:
    """
    Build the agentic workflow using StateGraph from LangGraph.
    Define the nodes, the edges and the input shared state and return compiled graph.

    Args:
        database_cfg: Configuration for the summary database.

    Returns:
        The compiled graph that can be executed when initial state is provided.
    """
    # set the summary database
    summary_database = SummaryDatabase(database_cfg)

    # set the LLM for the summary generator and critic nodes
    summary_generator = SummaryGenerator(node_cfg)

    graph = StateGraph(SharedState)

    # nodes
    graph.add_node("data gatherer", data_gatherer_node)

    # use `query_similar_summaries` function from summary database for database retrieval
    graph.add_node("memory retrieval", summary_database.query_similar_summaries)

    # add retry if exception raise when invoking the LLM (like timeout/connection error)
    graph.add_node(
        "summary generator", summary_generator, retry=RetryPolicy(max_attempts=3)
    )

    # edges
    graph.add_edge(START, "data gatherer")

    # if no news articles extracted, do not need to generate summaries and skip to end
    graph.add_conditional_edges(
        "data gatherer",
        check_gathered_data_present,
        {"end": END, "memory retrieval": "memory retrieval"},
    )
    graph.add_edge("memory retrieval", "summary generator")
    graph.add_edge("summary generator", END)

    return graph.compile()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> SharedState:
    """
    Main function that executes the agentic workflow (compiled graph) using empty initial state.
    Use LangFuse for observability where all traces are grouped under the section `financial_news_summary`.
    Each trace name contains the time where workflow is executed.

    Args:
        cfg: Configurations that are read using hydra.

    Returns:
        The final state of the graph after executing the workflow.
    """
    # get the configurations
    database_cfg = cfg.database
    node_cfg = cfg.node

    # get the compiled graph
    app = build_graph(database_cfg, node_cfg)

    # set initial state of graph where all fields are empty
    initial_state: SharedState = {
        "messages": [],
        "gathered_data": [],
        "past_summaries": [],
        "current_summary": [],
        "critic_feedback": [],
    }

    # get the current time of executing the graph to set the trace name
    SGT = ZoneInfo("Asia/Singapore")
    current_datetime = datetime.now(SGT)
    current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # set the Langfuse callback which traces every step of graph and send to LangFuse
    # read the LangFuse credentials from environment variables and make connection to LangFuse
    langfuse_handler = CallbackHandler()

    # set the established connection
    langfuse_client = get_client()

    # invoke the graph with the LangFuse
    # set the trace name (`run_name`) for this graph execution, which is includes date of execution
    # set the `session_id` under metadata to group all runs of the graph together in LangFuse
    # set a tag (date in ISO format) for easy reference
    result = app.invoke(
        initial_state,
        config={
            "callbacks": [langfuse_handler],
            "run_name": f"summary-{current_datetime_string}",
            "metadata": {
                "langfuse_session_id": "financial_news_summary",
                "langfuse_tags": [current_datetime.date().isoformat()],
            },
        },
    )

    # send all data to LangFuse before program exits
    langfuse_client.flush()

    # look at state of graph
    # print("Graph executed, current summary is :")
    # print(result["current_summary"])
    # print("Checking if summary is correct type:")
    # print(all(isinstance(item, ArticleSummary) for item in result["current_summary"]))
    # return the final state of the graph
    return result


if __name__ == "__main__":
    # return None by default
    main()
