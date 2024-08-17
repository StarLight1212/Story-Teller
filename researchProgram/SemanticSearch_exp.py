from typing import List, Tuple, Dict
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from StoryCore import StoryLLM
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from researchProgram.StoryOrigin import Story_Model
from langchain.memory import ConversationBufferMemory
from utils.conversation import conv_templates


class SemanticsFam_Expr(Story_Model):
    """
    Semantics Search Family for Experiments Protocol Retrieval
    """
    def __init__(self,
                 chat_model: StoryLLM,
                 ss_model,
                 ss_tokenizer,
                 pattern_pth: str,
                 embeddings_path: str = None,
                 embeddings_model: Embeddings = None,
                 persist_path: str = 'chroma_storage_exp',
                 ):
        super(SemanticsFam_Expr, self).__init__(chat_model, ss_model, ss_tokenizer, pattern_pth)
        # Load Embeddings Layers
        if embeddings_model is None and embeddings_path is not None:
            self.embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_path)
        elif embeddings_model is not None:
            self.embeddings = embeddings_model
        else:
            raise ValueError("embeddings_path or embeddings_model should be set first!")
        # Load Database
        self.load_vector_database(persist_path)
        # Setting Prompt Template
        self.template = self.normal_template()
        # Memory Mode
        self.memory = ConversationBufferMemory(
            input_key='question',
            memory_key="chat_history", return_messages=True)
        # Language Model Chain
        self.llm_chain = LLMChain(llm=self.chat_model, verbose=True,
                                    prompt=self.template,
                                    memory=self.memory)

    @staticmethod
    def normal_template():
        """Answer the question in fair, objective tone."""
        # Initial Context Templates
        template = """You are StoryLLM Chatbot having a conversation with a human.

        {chat_history}
        
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Please answer fairly, objectively and harmlessly."""
        return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

    def load_vector_database(self, persist_path) -> None:
        """Load the vector database from the disk."""
        self.vecdb = Chroma(persist_directory=persist_path,
                            embedding_function=self.embeddings)

    def generator_info(self, text):
        yield {"generated_text": text}

    def experiment_assistant(self,
                             query: str,
                             history: List[Dict] = None,
                             search_info_num: int = 4):
        # History Loading
        if history is not None:
            for msg in history:
                self.chat_model.conv.append_message(self.chat_model.conv.roles[0], msg['usr_input'])
                self.chat_model.conv.append_message(self.chat_model.conv.roles[1], msg['StoryLLM'])
        # Chat Settings
        # TODO: 当前无法支持不同对话间的查询
        history_pro = self.chat_model.conv.get_prompt()

        # Docs Retrievals with k related docs
        docs = self.vecdb.similarity_search_with_score(query, k=search_info_num)
        res = self.llm_chain.predict(context=docs, question=query, chat_history=history_pro)
        # add reference with answers
        answers_with_reference = res + '\n' + self.add_reference_format(docs)
        # Record users' QA history in current chat dialogue
        self.chat_model.conv.append_message(self.chat_model.conv.roles[0], query)
        self.chat_model.conv.append_message(self.chat_model.conv.roles[1], answers_with_reference)

        return self.generator_info(answers_with_reference)

    def add_reference_format(self, docs):
        suffix_ref = '[Reference]:\n'
        ref_set = set()
        if isinstance(docs, List):
            for i, doc in enumerate(docs):
                if isinstance(doc, Document):
                    ref_set.add(doc.metadata['title'] + '\nRef ' + doc.metadata['creator']
                                + '\nlink:' + doc.metadata['subject'] + '\n')
                elif isinstance(doc, Tuple):
                    doc = doc[0]
                    ref_set.add(doc.metadata['title'] + '\nRef ' + doc.metadata['creator']
                                + '\nlink:' + doc.metadata['subject'] + '\n')
            for i, item in enumerate(ref_set):
                suffix_ref += '[' + str(i + 1) + '] ' + item
            return suffix_ref

        elif isinstance(docs, Document):
            suffix_ref += '[1] ' + docs.metadata['title'] + '\nRef ' + docs.metadata['creator'] + '\nlink:' + \
                          docs.metadata[
                              'subject'] + '\n'
            return suffix_ref

        else:
            raise TypeError("Docs Param with the wrong type!")

    def clear_cache(self):
        del self.embeddings
