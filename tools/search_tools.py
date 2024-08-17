# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from typing import Optional, Type


# Load the tool configs that are needed.
search = SerpAPIWrapper()


class CustomSearchTool(BaseTool):
    """
    tools = [CustomSearchTool(), CustomCalculatorTool()]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
    """
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return search.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")