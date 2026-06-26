import hydra
from omegaconf import DictConfig
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from src.utils.logging import setup_logging
from src.state_and_dataclass import SharedState
from src.graph import build_graph

setup_logging()
logger = logging.getLogger("graph")

# load environment variable
load_dotenv()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> SharedState:
    """
    Main function that executes the agentic workflow (compiled graph) using empty initial state.
    Use LangFuse for observability where all traces are grouped under the section `financial_news_summary`.
    Each trace name contains the time when workflow is executed.

    Args:
        cfg: All configurations that are read using hydra.

    Returns:
        The final state of the graph after executing the workflow.
    """
    # get the configurations
    database_cfg = cfg.database
    node_cfg = cfg.node
    email_cfg = cfg.email

    # get the compiled graph
    app = build_graph(database_cfg, node_cfg, email_cfg)

    # save the graph diagram with conditional edges labeled
    # edges (start, end): label
    # edge_labels = {
    #     ("data gatherer",    "email sender"):      "no articles",
    #     ("data gatherer",    "memory retrieval"):  "articles extracted",
    #     ("summary generator","email sender"):      "threshold exceeded",
    #     ("summary generator","summary critic"):    "below threshold",
    #     ("summary critic",   "email sender"):      "approved",
    #     ("summary critic",   "summary generator"): "rejected",
    #     ("email sender",   "summary storage"):     "summaries present",
    #     ("email sender",   "__end__"):             "summaries absent",
    # }
    # # return node and edges of graph
    # drawn_graph = app.get_graph()

    # # replace conditional edge with edge with label
    # drawn_graph.edges = [
    #     edge._replace(data=edge_labels[(edge.source, edge.target)])
    #     if (edge.source, edge.target) in edge_labels
    #     else edge
    #     for edge in drawn_graph.edges
    # ]

    # png_bytes = drawn_graph.draw_mermaid_png()

    # with open("graph_diagram_complete.png", "wb") as f:
    #     f.write(png_bytes)
    #     logger.info("Graph diagram is drawn.")

    # set initial state of graph where all fields are empty
    # critic count is 0 and not yet approved
    initial_state: SharedState = {
        "messages": [],
        "gathered_data": [],
        "past_summaries": [],
        "current_summary": [],
        "critic_feedback": [],
        "approved": False,
        "critic_count": 0,
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

    logger.info("Graph execution completed.")

    # send all data to LangFuse before program exits
    langfuse_client.flush()

    # look at state of graph
    # print("Graph executed, the final state is:")
    # print(f"Number of current summary: {len(result['current_summary'])}")
    # for i in result["current_summary"]:
    #     print(f"title: {i.title}\n")
    #     print(f"content: {i.content}\n")

    # print(f"Number of critic feedback: {len(result['critic_feedback'])}")
    # for i in result["critic_feedback"]:
    #     print(i)
    #     print("\n")

    # print(f"approval status: {result['approved']}\n")
    # print(f"critic count: {result['critic_count']}\n")

    return result


if __name__ == "__main__":
    main()
