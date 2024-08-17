from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun


class SumInput(BaseModel):
    question: str = Field()
    history: str = Field()


class SummarizeTool(BaseTool):
    name = "Basic Summarization"
    description = "useful for when you need to summarize the document"
    args_schema: Type[BaseModel] = SumInput

    def __init__(self, llm: LLM):
        super(SummarizeTool, self).__init__()
        self.summarize_chain = load_summarize_chain(llm=llm, chain_type='map_reduce')

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.summarize_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")


class QA_SummarizeTool(BaseTool):
    name = "QA Summarization"
    description = "useful for when you need to summarize the document"
    args_schema: Type[BaseModel] = SumInput

    def __init__(self, llm: LLM):
        super(QA_SummarizeTool, self).__init__()
        self.summarize_chain = load_qa_chain(llm=llm, chain_type='map_reduce')

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.summarize_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")

