import feedparser
import trafilatura  # one library to extract news article content
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging

from src.utils.logging import setup_logging
from src.node.state_and_dataclass import SharedState, Article

setup_logging()
logger = logging.getLogger("function_node")


def convert_pubdate_to_specific_timezone(
    pubdate_str: str, timezone: ZoneInfo
) -> datetime.datetime:
    """
    Convert the published date (in RSS string format) of the news article
    to datetime in specific time zone for accurate comparison.

    Args:
        pubdate_str: Published date string used in RSS.
        timezone: Time zone to convert the time to.

    Returns:
        The publised date in timezone-aware datetime.
    """
    dt = parsedate_to_datetime(pubdate_str)
    dt_timezone = dt.astimezone(timezone)
    return dt_timezone


def extract_article_text(input_url: str) -> str:
    """
    Use trafilatura library to download the raw HTML content of a webpage from the input_url
    and extracts the main article text, stripping away boilerplate like ads, navbars, and footer
    which are common on webpages and ignore comments and tables contents.

    Args:
        input_url: The URL of the news article to extract the text from.

    Returns:
        The text of the news article if url page can be fetched and text is extracted, else an empty string.
    """
    if input_url:
        downloaded = trafilatura.fetch_url(input_url)
        if not downloaded:
            logger.info(f"Error fetching contents from url: {input_url}")
            return ""

        # extract text from the html extracted
        article_text = trafilatura.extract(
            downloaded, include_comments=False, include_tables=False
        )
        # can return None if extraction fails
        return article_text.strip() if article_text else ""
    else:
        return ""


def data_gatherer_node(state: SharedState) -> dict:
    """
    Fetch up to a max number of financial news articles published in the last 24 hours
    from predefined RSS feed URLs, and populates the `gathered_data` field in shared state.
    Each news article extracted is converted and stored as an `Article` object.

    Args:
        state: Common graph state across the graph.

    Returns:
        Updated `gathered_data` field in the shared state.
    """
    # the RSS feed url for the specific websites to get news articles from
    RSS_FEEDS = {
        "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "CNA Business": "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6936",
    }

    # set maximum article to extract per RSS url
    MAX_ARTICLE_PER_FEED = 10

    # want the past one day news articles
    # set the look back range to be from yesterday
    SGT = ZoneInfo("Asia/Singapore")
    yesterday_date_sgt = datetime.now(SGT).date() - timedelta(days=1)
    logger.info(f"End date of look back range for news article: {yesterday_date_sgt}")

    # store all articles
    articles = []

    for source, feed_url in RSS_FEEDS.items():
        logger.info(f"Extracting RSS feeds from source: {source}")
        # read the RSS feed
        parsed = feedparser.parse(feed_url)
        counter = 0

        for entry in parsed.entries:
            if counter < MAX_ARTICLE_PER_FEED:
                # get published date of article
                published_date = getattr(entry, "published", "").strip()

                # if date present, check if it is within look back range and convert date to ISO format
                # else, skip the conversion
                # break the for loop when all articles are beyond look back range
                if published_date:
                    # convert to SGT time
                    published_date_sgt = convert_pubdate_to_specific_timezone(
                        published_date, SGT
                    )
                    # check within look back range
                    if published_date_sgt.date() >= yesterday_date_sgt:
                        # convert to ISO format
                        published_date = published_date_sgt.isoformat()
                    else:
                        # articles in chronological order, any articles after will be more dated
                        logger.info(
                            "Breaking RSS feed for loop since all articles are beyond look back range."
                        )
                        break
                # extract link, title and article content
                link = getattr(entry, "link", "").strip()
                title = getattr(entry, "title", "").strip()
                content = extract_article_text(link)
                articles.append(
                    Article(
                        source=source,
                        title=title,
                        published_date=published_date,
                        link=link,
                        content=content,
                    )
                )
                counter += 1
            else:
                logger.info("Breaking RSS feed for loop since max articles reached.")
                break

    # update the state
    return {"gathered_data": articles}
