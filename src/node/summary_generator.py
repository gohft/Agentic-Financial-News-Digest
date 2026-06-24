from omegaconf import DictConfig, OmegaConf
import logging
from typing import List
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from src.utils.logging import setup_logging
from src.utils.llm_helper import get_llm
from src.node.state_and_dataclass import SharedState, ArticleSummaryList

setup_logging()
logger = logging.getLogger("llm_node")


class SummaryGenerator:
    """
    Use LLM model to generate summary for the significant news article,
    ensuring that the summary does not repeat information already covered in previous summaries.
    """

    def __init__(self, cfg: DictConfig):
        """
        Set the LLM to use to generated the summaries from the extracted news articles and
        past summaries using the configuration file that contains the model arguments.

        LLM will output the generated summaries as a structured output so as to ensure the
        summaries can be added into the vector database.
        """
        # get config for the summary generator
        llm_cfg = cfg["generator"]

        # get system prompt
        self.system_prompt = llm_cfg["system_prompt"]

        # get llm provider
        self.provider = llm_cfg["provider"]

        # use the provider and model arguments to set the llm to use
        # get the model arguments and convert to dictionary type
        # model arguments must have keys that match the input arguments of the model
        self.model_args = OmegaConf.to_container(llm_cfg["model_args"])

        try:
            self.llm = get_llm(self.provider, self.model_args)

            # set the model to output a the object of type ArticleSummaryList (list type not allowed)
            # "json_mode" allow LLM to output in JSON format (following system promt) before parsing to ArticleSummaryList
            # ArticleSummary object type required to insert the summary into database
            # no tools assigned to LLM since the task is to do summarization and all information provided
            self.structured_llm = self.llm.with_structured_output(
                ArticleSummaryList, method="json_mode"
            )
            logger.info("Set the LLM for summary generation.")
        except ValueError as e:
            logger.error(f"Failure to set LLM to generate summaries: {e}")
            raise

    def _create_user_prompt(self, state: SharedState) -> str:
        """
        Create the user prompt for the model.
        User request contains the all the information to generate the summaries, including
        the extracted news articles, the past summaries and the critic feedback.
        """

        # get all the necessary information needed to generate the summaries
        news_articles = state.get("gathered_data")
        similar_past_summaries = state.get("past_summaries", [])
        summary_feedback = state.get("critic_feedback", [])

        # title and content of each news article separated by a line
        articles_block = "\n\n".join(
            f"Title: {article.title}\nContent: {article.content}"
            for article in news_articles
        )

        # past summaries can be empty list, set to None following LLM system prompt
        past_summaries_block = "None"
        if similar_past_summaries:
            past_summaries_block = "\n\n".join(
                f"Title: {summary.title}\nContent: {summary.content}"
                for summary in similar_past_summaries
            )
            logger.info("Added similar past summaries into user prompt")

        # feedback can be empty list, set to None following LLM system prompt
        feedback_block = "None"
        if summary_feedback:
            feedback_block = "\n\n".join(
                f"Title: {feedback.title}\nComments: {feedback.comments}"
                for feedback in summary_feedback
            )
            logger.info("Added feedback from summary critic into user prompt")

        return (
            f"=== GATHERED ARTICLES ===\n{articles_block}\n\n"
            f"=== PREVIOUS DIGEST ===\n{past_summaries_block}\n\n"
            f"=== CRITIC FEEDBACK ===\n{feedback_block}\n\n"
        )

    def __call__(self, state: SharedState) -> dict:
        """
        Set the class to be callable.
        Invoke the LLM to generate summary when the class is called.

        To generate summary, start a new conversation with LLM, ignoring previous history.
        This means not passing `messages` from SharedState into the LLM.

        New conversation prevents too long context window and restates the full news article,
        this reduces the chance of exceeding context window and hallucination.
        This forces model to generate new summaries instead of correcting previous summaries,
        as multiple corrections could lead to model arguing/explaining with itself.
        """
        # create the user request
        user_prompt = self._create_user_prompt(state)

        # do not pass messages from shared state
        # start new conversation by passing only system prompt and user request
        messages: List[AnyMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # return structured output in List[ArticleSummary] and not AnyMessage type
        # do not need to append to messages since this is not part of conversation
        # and do not need to repeat since store in `current_summary` in state
        generated_summaries = self.structured_llm.invoke(messages)

        logger.info("Summaries generated by summary generator.")

        return {
            "messages": messages,  # record each of the user input
            "current_summary": generated_summaries.articles,  # get the list
        }
