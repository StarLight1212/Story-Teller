from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from typing import Optional, Type, List, Dict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains import ConversationalRetrievalChain


class QATool(BaseTool):
    name = "Question Answering Tool"
    description = "useful for when you need to answer questions with documents"

    def __init__(self, llm: LLM):
        super(QATool, self).__init__()
        self.summarize_chain = load_qa_chain(llm=llm, chain_type='stuff')

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.summarize_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")


class ConvRetrievalTool(BaseTool):
    name = "Question Answering Tool"
    description = "useful for when you need to answer questions with documents"

    def __init__(self, llm: LLM,
                 vector_store,
                 with_history: bool = False):
        super(ConvRetrievalTool, self).__init__()
        if with_history:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            self.summarize_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        else:
            self.summarize_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

    def _run(self, query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None,
             history: Optional[List[Dict]] = None) -> Dict:
        """Use the tool."""
        return self.summarize_chain({"question": query, "chat_history": history})

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")
