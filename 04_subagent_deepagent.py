from dotenv import load_dotenv
import os

from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
load_dotenv()

os.environ["OLLAMA_API_KEY"]=os.getenv("OLLAMA_API_KEY")
os.environ["OLLAMA_BASE_URL"]=os.getenv("OLLAMA_BASE_URL")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")



tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""
model = init_chat_model(model="ollama:llama3.1:8b", base_url=os.environ["OLLAMA_BASE_URL"], api_key=os.environ["OLLAMA_API_KEY"])

research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": model,  # Optional override, defaults to main agent model
}
subagents = [research_subagent]



agent = create_deep_agent(
    model=model,
    subagents=subagents,
    debug=True,
)

# result = agent.invoke({"messages": [{"role": "user", "content": "What is hermitian matrix?"}]})
result = agent.invoke({"messages": [{"role": "user", "content": "What is orthogonal matrix?"}]})

# Print the agent's response
print(result["messages"][-1].content)

print("Done!")