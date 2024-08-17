from langchain import LLMMathChain
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun


class CalculatorInput(BaseModel):
    question: str = Field()


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def __init__(self, llm: LLM):
        super(CustomCalculatorTool, self).__init__()
        self.llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.llm_math_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")