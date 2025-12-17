from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
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



