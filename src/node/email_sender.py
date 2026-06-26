import logging
from omegaconf import DictConfig
import smtplib
from datetime import datetime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText

from src.utils.logging import setup_logging
from src.state_and_dataclass import SharedState

setup_logging()
logger = logging.getLogger("function_node")


class EmailSender:
    """
    Email the outcome of the workflow to user using gmail smtp.

    If there is any complete summary (non-empty title and content) generated.
    email the summary (title + content) in order to the user.

    If there is no complete summary or no summary generated at all,
    email to inform user that there is no summary generated.

    If there is no news article extracted, email to inform user about it.

    Include retries in sending email with max retry set.
    """

    # define the default HTML template and summary block template
    HTML_TEMPLATE = """\
        <!DOCTYPE html>
        <html lang="en">
        <body>
        <div class="body">
        {summaries}
        </div>
        <div class="footer">
        This email was generated automatically. Please do not reply.
        </div>
        </body>
        </html>
        """
    # title is bold and larger
    # content is under title
    SUMMARY_BLOCK = """\
        <div>
        <p style="font-size: 18px; font-weight: bold;">{title}</p>
        <p style="font-size: 14px;">{content}</p>
        <br>
        </div>
        """

    # set default message
    NO_ARTICLE_MSG = "No news articles were extracted for summary."
    NO_SUMMARY_MSG = "No summaries generated from extracted news articles"

    def __init__(self, cfg: DictConfig) -> None:
        """
        Set the configurations to email user using gmail SMTP.

        Args:
            cfg: Configurations for the email sender.
        """

        self.smtp_password = cfg["smtp_password"]
        self.sender = cfg["sender"]
        self.recipient = cfg["recipient"]
        self.smtp_port = cfg["port"]

        self.smtp_host = "smtp.gmail.com"

        self.max_retries = cfg["max_retries"]

    def create_email_message(self, state: SharedState) -> str:
        """
        Using the `gathered_data` and `current_summary` in state to create
        the email message to send.

        If no articles extracted, return `NO_ARTICLE_MSG`.
        If no complete summaries generated, return `NO_SUMMARY_MSG`
        If there is complete summary, create SUMMARY_BLOCKs for each summary
        and concantenate them into a long string to input into `HTML_TEMPLATE`.

        Args:
            state: Common graph state across the graph.

        Returns:
            String with default message or string with summaries inputted into the
            HTML_TEMPLATE.
        """
        # get gathered data
        news_articles_extracted = state.get("gathered_data")

        # get current summary
        generated_summaries = state.get("current_summary")

        # no news article extracted
        if not news_articles_extracted:
            logger.info("No articles extracted, set email message as string.")
            return self.NO_ARTICLE_MSG

        # no summaries generated
        if not generated_summaries:
            logger.info("No summaries generated, set email message as string.")
            return self.NO_SUMMARY_MSG

        # filter for summaries that have title and content non empty string
        complete_summaries = [
            sm for sm in generated_summaries if sm.title.strip() and sm.content.strip()
        ]

        # concatenate all summary blocks together for long string
        if complete_summaries:
            # concatenate all summary blocks together for long string
            summary_blocks = "".join(
                self.SUMMARY_BLOCK.format(title=sm.title, content=sm.content)
                for sm in complete_summaries
            )
            logger.info("HTML email message created using generated summaries.")
            return self.HTML_TEMPLATE.format(summaries=summary_blocks)
        else:
            logger.info("No summaries generated, set email message as string.")
            return self.NO_SUMMARY_MSG

    def send_email(self, state: SharedState) -> None:
        """
        Create the email message and email user over secured SMTP connection with retries.

        Raises:
            RuntimeError if fail to send email after max retries.
        """
        body = self.create_email_message(state)

        # set the subject to current sent time
        SGT = ZoneInfo("Asia/Singapore")
        current_datetime = datetime.now(SGT)
        current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        subject = f"Daily Financial Digest {current_datetime_string} SGT"

        if body in [self.NO_SUMMARY_MSG, self.NO_ARTICLE_MSG]:
            # body is plain string
            msg = MIMEText(body, "plain")
        else:
            # body is html string
            msg = MIMEText(body, "html")

        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = self.recipient

        last_exception = None
        # send email with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                # Opens a TCP connection to smtp.gmail.com on provided port
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    # send handshake to the SMTP server
                    server.ehlo()
                    # upgrades connection to encrypted TLS connection
                    server.starttls()
                    # authenticate sender
                    server.login(self.sender, self.smtp_password)
                    # send email
                    server.sendmail(self.sender, self.recipient, msg.as_string())
                logger.info(f"Email sent: {subject}")
                return
            except Exception as e:
                # set the lastest exception
                last_exception = e
                # reach max attempts
                if attempt == self.max_retries:
                    logger.info(
                        f"Error sending email at attempt {attempt}, max retries reached, no more retries."
                    )
                    logger.info(f"Error: {e}")
                else:
                    logger.info(
                        f"Error sending email at attempt {attempt}, retrying again."
                    )
                    logger.info(f"Error: {e}")

        raise RuntimeError(
            f"Failed to send email after {self.max_retries} attempts."
        ) from last_exception


# test
# import hydra
# import time
# from dotenv import load_dotenv
# from src.state_and_dataclass import ArticleSummary, Article

# # load environment variable
# load_dotenv()
# @hydra.main(config_path="../../config", config_name="config", version_base=None)
# def main(cfg: DictConfig) -> None:
#     # get the configurations
#     email_cfg = cfg.email

#     email_sender = EmailSender(email_cfg)

#     # test empty
#     # test with empty string
#     # test with 2 ArticleSummary

#     no_article_state = {
#         "messages": [],
#         "gathered_data": [],
#         "past_summaries": [],
#         "current_summary": [],
#         "critic_feedback": [],
#         "approved": False,
#         "critic_count": 0,
#     }

#     no_summary_state = {
#         "messages": [],
#         "gathered_data": [Article(source="test", title="test", published_date="test", link="test", content="test")],
#         "past_summaries": [],
#         "current_summary": [],
#         "critic_feedback": [],
#         "approved": False,
#         "critic_count": 0,
#     }

#     empty_string_state = {
#         "messages": [],
#         "gathered_data": [Article(source="test", title="test", published_date="test", link="test", content="test")],
#         "past_summaries": [],
#         "current_summary": [ArticleSummary(title="test", content="")],
#         "critic_feedback": [],
#         "approved": False,
#         "critic_count": 0,
#     }

#     two_summary_state = {
#         "messages": [],
#         "gathered_data": [Article(source="test", title="test", published_date="test", link="test", content="test")],
#         "past_summaries": [],
#         "current_summary": [ArticleSummary(title="title 1", content="content1"), ArticleSummary(title="title 2", content="content 2")],
#         "critic_feedback": [],
#         "approved": False,
#         "critic_count": 0,
#     }

#     #all_states = [no_article_state, no_summary_state, empty_string_state, two_summary_state]
#     all_states = [empty_string_state]
#     for st in all_states:
#         time.sleep(5)
#         email_sender.send_email(st)

# if __name__ == "__main__":
#     main()
