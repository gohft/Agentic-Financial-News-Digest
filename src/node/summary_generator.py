from omegaconf import DictConfig, OmegaConf
import logging
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from src.utils.logging import setup_logging
from src.utils.llm_helper import get_llm
from src.state_and_dataclass import SharedState, ArticleSummaryList, ArticleSummary

setup_logging()
logger = logging.getLogger("llm_node")


class SummaryGenerator:
    """
    Use LLM model to generate summary for each significant news article,
    among all extracted news article, ensuring that the summary does not repeat
    information already covered in previous summaries.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Set the LLM to use to generated the summaries from the extracted news articles and
        past summaries using the configuration file that contains the model arguments.

        LLM will output the generated summaries as a structured output so as to ensure the
        summaries can be added into the vector database.

        Args:
            cfg: Configurations for the summary generator LLM.

        Raise:
            ValueError if LLM cannot be initialized when the provider is not in allowed list.
        """
        # get config for the summary generator
        llm_cfg = cfg["generator"]

        # get system prompt
        self.system_prompt = llm_cfg["system_prompt"]

        # get llm provider
        self.provider = llm_cfg["provider"]

        # convert model arguments from config to dictionary type
        # model arguments must have keys that match the input arguments of the model
        self.model_args = OmegaConf.to_container(llm_cfg["model_args"])

        try:
            self.llm = get_llm(self.provider, self.model_args)

            # set the model to output object of type ArticleSummaryList (list type not allowed)
            # "json_mode" allow LLM to first output in JSON format (following system promt)
            # before parsing to ArticleSummaryList, reducing chances of mistakes
            # no tools assigned to LLM since the task is to do summarization and all information provided
            self.structured_llm = self.llm.with_structured_output(
                ArticleSummaryList, method="json_mode"
            )
            logger.info("Set the LLM for summary generation.")
        except ValueError as e:
            logger.error(f"Failure to set LLM to generate summaries: {e}")
            raise

    @staticmethod
    def _create_user_prompt(state: SharedState) -> str:
        """
        Create the user prompt for the model.
        User request contains the all the information to generate the summaries, including
        the extracted news articles, the past summaries and the critic feedback.

        Args:
            state: Common graph state across the graph.

        Returns:
            The HumanMessage that will invoke response from LLM.
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
            logger.info(
                "Added similar past summaries into summary generator user prompt"
            )

        # feedback can be empty list, set to None following LLM system prompt
        feedback_block = "None"
        if summary_feedback:
            feedback_block = "\n\n".join(
                f"Title: {feedback.title}\nComments: {feedback.comments}"
                for feedback in summary_feedback
            )
            logger.info(
                "Added feedback from summary critic into summary generator user prompt"
            )

        logger.info("Created summary generator user prompt")

        # output format matches the input stated in LLM system prompt
        return (
            f"=== GATHERED ARTICLES ===\n{articles_block}\n\n"
            f"=== PREVIOUS DIGEST ===\n{past_summaries_block}\n\n"
            f"=== CRITIC FEEDBACK ===\n{feedback_block}\n\n"
        )

    def __call__(
        self, state: SharedState
    ) -> dict[str, list[AnyMessage] | list[ArticleSummary]]:
        """
        Set the class to be callable.
        Invoke the LLM to generate summary when the class is called.

        To generate summary, start a new conversation with LLM, ignoring previous history.
        This means not passing `messages` from SharedState into the LLM.

        New conversation prevents too long context window and restates the full news article,
        this reduces the chance of exceeding context window and hallucination.
        This forces model to generate new summaries instead of correcting previous summaries,
        as multiple corrections could lead to model arguing/explaining with itself.

        Each generated summary is of type ArticleSummary so as to be added to vector database.
        Filter to get at max the first 5 objects in list, since LLM might not follow prompt
        and output more than 5 summaries, but expected only max of 5 summaries in daily digest.

        Args:
            state: Common graph state across the graph.

        Returns:
            Updated `messages` and `current_summary` field in the shared state.
        """
        # create the user request
        user_prompt = self._create_user_prompt(state)

        # do not pass messages from shared state
        # start new conversation by passing only system prompt and user request
        messages: list[AnyMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # return structured output in list[ArticleSummary] and not AnyMessage type
        # do not need to append to messages since this is not part of conversation
        generated_summaries = self.structured_llm.invoke(messages)

        logger.info("Summaries generated by summary generator.")
        logger.info(
            f"Number of summaries generated is: {len(generated_summaries.articles)}"
        )

        # update the state with the newly generated summaries and add the LLM prompts
        return {
            "messages": messages,  # record each of the user input
            "current_summary": generated_summaries.articles[
                :5
            ],  # filter to get max of 5
        }
