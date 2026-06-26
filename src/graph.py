from omegaconf import DictConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
import logging

from src.utils.logging import setup_logging
from src.state_and_dataclass import SharedState
from src.node.data_gatherer import data_gatherer_node
from src.node.memory_retrieval import SummaryDatabase
from src.node.summary_generator import SummaryGenerator
from src.node.summary_critic import SummaryCritic

setup_logging()
logger = logging.getLogger("graph")


#########################
# conditional edges logic
#########################

# Email node and end


# EDIT to direct to email instead of to END
def check_gathered_data_present(state: SharedState) -> str:
    """
    If there are news articles extracted after the `data gatherer` node,
    move to `memory retrieval` node to get past summaries that have similar
    content as the articles, else skip directly to end of graph.

    Args:
        state: Common graph state across the graph.

    Returns:
        The name of the next node (`end` or `memory retrieval`) from `data gatherer` node.
    """
    if not state["gathered_data"]:
        logger.info("No articles extracted from websites, ending the graph.")
        return "end"

    logger.info("Articles extracted from websites, moving to memory retrieval node.")
    return "memory retrieval"


def check_critic_count_exceeded(state: SharedState, threshold: int = 2) -> str:
    """
    If the `critic_count` in state has reached the threshold after the `summary generator` node,
    skip the `summary critic` node and move to `summary storage` node to store the summaries,
    else move to `summary critic` node for summary evaluation.

    Args:
        state: Common graph state across the graph.
        threshold: The critic count threshold that when reached, will skip the critic step.
                Default is 2.

    Returns:
        The name of the next node (`summary storage` or `summary critic`) from `summary generator` node.
    """
    critic_count = state["critic_count"]
    # reach threshold
    if critic_count >= threshold:
        logger.info(
            f"Critic count: {critic_count} exceeded critic threshold: {threshold}, skipping critic node, moving to storage node."
        )
        return "summary storage"

    logger.info(
        f"Critic count: {critic_count} below critic threshold: {threshold}, moving to critic node."
    )
    return "summary critic"


def check_summary_approval_status(state: SharedState) -> str:
    """
    If the `approved` in state is True after the `summary critic` node,
    move to `summary storage` node to store the summaries, else move to
    `summary generator` node for regenerate the summaries following the
    feedback from the critic.

    Args:
        state: Common graph state across the graph.

    Returns:
        The name of the next node (`summary storage` or `summary generator`) from `summary critic` node.
    """
    approval_status = state["approved"]
    # if approved
    if approval_status:
        logger.info("Summary approved by critic, moving to storage node.")
        return "summary storage"

    # not approved, regenerate
    logger.info(
        "Summary not approved by critic, moving back to generator for regeneration."
    )
    return "summary generator"


#######
# Graph
#######
def build_graph(database_cfg: DictConfig, node_cfg: DictConfig) -> CompiledStateGraph:
    """
    Build the agentic workflow graph containing the `SharedState` using LangGraph.
    Define the nodes, the edges and the edges' logic and return the compiled graph.

    Args:
        database_cfg: Configuration for the vector database to use to store past summaries.
        node_cfg: Configurations for the summary generator and summary critic LLM.

    Returns:
        The compiled graph that can be executed when initial state is provided.
    """
    # set summary database
    summary_database = SummaryDatabase(database_cfg)

    # clear database
    # summary_database.empty_data_in_collection()

    # set LLM for the summary generator and critic nodes
    summary_generator = SummaryGenerator(node_cfg)
    summary_critic = SummaryCritic(node_cfg)

    # create a state graph
    graph = StateGraph(SharedState)

    # define the nodes
    graph.add_node("data gatherer", data_gatherer_node)

    # use `query_similar_summaries` function from summary database for database retrieval
    graph.add_node("memory retrieval", summary_database.query_similar_summaries)

    # add retry if exception raise when invoking the LLM (like timeout/connection error)
    graph.add_node(
        "summary generator", summary_generator, retry=RetryPolicy(max_attempts=3)
    )

    graph.add_node(
        "summary critic",
        summary_critic.feedback_and_get_approval_status,
        retry=RetryPolicy(max_attempts=3),
    )

    # use `add_summaries` function from summary database to add summaries generated
    graph.add_node("summary storage", summary_database.add_summaries)

    # add the edges and logic
    graph.add_edge(START, "data gatherer")

    # conditional edge from data gatherer, depend if articles extracted or not
    graph.add_conditional_edges(
        "data gatherer",
        check_gathered_data_present,
        {"end": END, "memory retrieval": "memory retrieval"},
    )
    graph.add_edge("memory retrieval", "summary generator")

    # conditional edge from summary generator, depend on critic_count value and threshold
    graph.add_conditional_edges(
        "summary generator",
        check_critic_count_exceeded,
        {"summary storage": "summary storage", "summary critic": "summary critic"},
    )

    # conditional edge from summary critic, depend on approval status after feedback
    graph.add_conditional_edges(
        "summary critic",
        check_summary_approval_status,
        {
            "summary storage": "summary storage",
            "summary generator": "summary generator",
        },
    )

    graph.add_edge("summary storage", END)

    logger.info("Graph is built and compiled.")

    return graph.compile()
