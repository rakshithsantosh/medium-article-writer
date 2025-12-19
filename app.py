from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.youtube import YouTubeTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.x import XTools
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# define the path to save research papers
dir_path = Path("./research_papers")

dir_path.mkdir( exist_ok=True)

# define orchestrator and agent model

orchestrator = OpenAIChat(model="gpt-4")

model = OpenAIChat(model="gpt-4.1-mini")

# define my sub agents / members

# define the arxiv research agent
arxiv_research_agent = Agent(
    id="arxiv_research_agent",
    name="Arxiv Research Agent",
    model=model,
    role="Arxiv Research Assistant",
    instructions="You are an expert research assistant specialized in finding and summarizing academic papers. Use the ArxivTools to search for papers based on user queries and provide concise summaries of the findings.",
    tools=[ArxivTools(download_dir=dir_path)],
    add_datetime_to_context=True,
)

# define the websearch agent
websearch_agent = Agent(
    id="websearch_agent",
    name="Web Search Agent",
    model=model,
    role="Web Search Assistant",
    instructions="You are an expert research assistant specialized in finding information from the web. Use web search tools to find relevant articles and summarize the key points for the user.and also provide links to the sources you used.",
    tools=[DuckDuckGoTools(), GoogleSearchTools()],
    add_datetime_to_context=True,
)

# define the hacker news research agent
hackernews_research_agent = Agent(
    id="hackernews_research_agent",
    name="Hacker News Research Agent",
    model=model,
    role="Hacker News Research Assistant",
    instructions="You are an expert research assistant specialized in finding and summarizing information from Hacker News. Get relevant information about the articles for the topic the user requested for,summarize your finding in proper format",
    tools=[HackerNewsTools()],
    add_datetime_to_context=True,
)

# define the news article research agent

news_article_research_agent = Agent(
    id="news_article_research_agent",
    name="News Article Research Agent",
    model=model,
    role="News Article Research Assistant",
    role="News Article Research Assistant",
    instructions="You are a research assistant that can read the contents of articles, whenever url is provided you can read the content of the article and also get its data using the available tools search for articles and summarize them and gather relevant information.",
    tools=[Newspaper4kTools(include_summary=True)],
    add_datetime_to_context=True,
)

# define the wikipedia tool
wikipedia_research_agent = Agent(
    id="wikipedia_research_agent",
    name="Wikipedia Research Agent",
    model=model,
    role="Wikipedia Research Assistant",
    instructions="You are an expert research assistant specialized in finding and summarizing information from Wikipedia. Use the Wikipedia tool to gather relevant information about the topic the user requested for, sum marize your findings in a proper format.",
    tools=[WikipediaTools()],
    add_datetime_to_context=True,
)

# define x research agent

#x_research_agent = Agent(
    #id="x_research_agent",
    #name="X Research Agent",
    #model=model,
    #role="X Research Assistant",
    #instructions="You are a research assistant that gather information from X formarly known as Twitter. You can search for relevant tweets and summarize the key points for the user.Do include the metric information to the posts you are referring to.Summarize your research in clear and concise manner.",
    #tools=[XTools(include_post_metrics=True,wait_on_rate_limit=True)],
    #add_datetime_to_context=True,
#)

# define youtube research agent

youtube_research_agent = Agent(
    id="youtube_research_agent",
    name="YouTube Research Agent",
    model=model,
    role="YouTube Research Assistant",
    instructions="You are a research assistant that can find relevant YouTube videos based on user queries. Use the available tools to search for videos, gather information from the video transcripts,You can also read metadata of the video and summarize the key points for the user.",
    tools=[YouTubeTools()],
    add_datetime_to_context=True,
)



