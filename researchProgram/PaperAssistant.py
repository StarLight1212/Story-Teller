import os
from typing import List, Generator
from utils.fewshot_prompt import ai_response_for_summary, usr_for_summary
from langchain.schema import Document
from weResearchCore import WeResearch
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings.base import Embeddings
from langchain.document_loaders import ArxivLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain
from researchProgram.weResearchOrigin import WeResearch_Model


class WeResearch_PaperAss(WeResearch_Model):
    """
    WeResearch Paper Assistant with PDF docs
    """
    tools: List = []
    user_demo: List = usr_for_summary
    ai_demo: List = ai_response_for_summary

    def __init__(self,
                 chat_model: WeResearch,
                 ss_model,
                 ss_tokenizer,
                 pattern_pth: str,
                 embeddings_path: str = None,
                 embeddings_model: Embeddings = None,
                 ):
        super(WeResearch_PaperAss, self).__init__(chat_model, ss_model, ss_tokenizer, pattern_pth)
        # Load Embeddings Layers
        if embeddings_model is None and embeddings_path is not None:
            self.embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_path)
        elif embeddings_model is not None:
            self.embeddings = embeddings_model
        else:
            raise ValueError("embeddings_path or embeddings_model should be set first!")
        # Loading Few shot demo into history
        self.few_show_setting(self.user_demo, self.ai_demo)
        # Summarization
        # self.sum_tools = SummarizeTool(llm=self.chat_model)
        # self.tools.append(self.sum_tools)
        prompt = self.get_prompt()
        self.llm_chain = LLMChain(llm=self.chat_model, verbose=True, prompt=prompt)

    def load_pdf(self, pth: str):
        """Load PDF File Embeddings"""
        return PyMuPDFLoader(pth).load()

    def get_prompt(self):
        # Initial Context Templates
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer fairly, objectively and harmlessly in English:"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return prompt

    def search_from_arxiv(self,
                          query: str,
                          load_max_docs: int = 3) -> List[Document]:
        """
        Loads a query result from arxiv.org into a list of Documents.

        Each document represents one Document.
        The loader converts the original PDF format into the text.
        :param query:
        :param load_max_docs:
        :return: Document()
        """
        docs = ArxivLoader(query=query, load_max_docs=load_max_docs)
        return docs.load()

    def construct_docs_from_pdf(self, file):
        # 将文件保存到 media 文件夹下的临时文件
        temp_file_path = os.path.join('./tmp/', file['filename'])
        with open(temp_file_path, 'wb') as temp:
            temp.write(file['body'])

        try:
            # Loading PDF from the file path
            docs = self.load_pdf(temp_file_path)
            # Splitting Documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=486, chunk_overlap=20)
            split_docs = text_splitter.split_documents(docs)

        finally:
            # Remove Temp File
            os.remove(temp_file_path)

        return split_docs

    @staticmethod
    def query_from_chroma(vec_db: Chroma, hash_token: str):
        """
        :param vec_db: vector database
        :param hash_token: each hash token correlated with each unique documents
        :return: if didn't find, then return the dict of: {'ids': [], 'embeddings': [], 'documents': [], 'metadatas': []}
        """
        search_results = vec_db.get(
            # Filter with hash token
            where={
                "hash value": {
                    "$eq": hash_token
                }
            },
            # Set the information that u want to return
            include=["embeddings", "documents", "metadatas"]
        )

        return search_results

    def load_from_urls(self, urls: List[str]):
        """
        parse the url content
        """
        loader = UnstructuredURLLoader(urls=urls)
        return loader.load()

    def load_vec_db(self, file, vector_db: Chroma, hash_token: str):
        # 1. Discriminate whether the pdf docs have been exited
        # Search with hash value
        search_results = self.query_from_chroma(vector_db, hash_token)
        # 2. IF HASH VALUE NOT FOUND IN DATABASE"
        if not search_results['documents']:
            # Splitting docs
            split_docs = self.construct_docs_from_pdf(file)
            # Documents to List[str]
            texts = [doc.page_content for doc in split_docs]
            # Load meta data from split documents
            metadatas = [doc.metadata for doc in split_docs]
            # Record hash value into each files' meta_data
            for meta in metadatas:
                # add hash token
                meta['hash value'] = hash_token
            # Adding into Chroma Database
            # Construct vector database with current docs
            vector_db.add_texts(texts,
                                metadatas=metadatas
                                # ids=[hash_token]*len(texts)
                                )
            # Persistence Restore
            vector_db.persist()
            # Search for the current docs information again
            search_results = self.query_from_chroma(vector_db, hash_token)
        # 3. NOW THE HASH VALUE MUST COULD BE FOUND IN DATABASE"
        get_docs = search_results['documents']
        # Construct temp Chroma Database
        tmp_vec_db = Chroma.from_texts(texts=get_docs, embedding=self.embeddings)

        return tmp_vec_db

    def generate_hold(self, text):
        yield {"generated_text": text}

    def few_show_setting(self, usr_r_lst: List, ai_r_lst: List):
        """Setting Few-shot demo into Conversation class"""
        # Assert the length of user demo and ai demo is the same value
        assert len(usr_r_lst) == len(ai_r_lst)
        for i in range(len(usr_r_lst)):
            self.chat_model.conv.set_conv_history(usr_r_lst[i], ai_r_lst[i])

    def chat_with_docs(self,
                       query: str,
                       vector_db: Chroma = None
                       ) -> Generator:
        """
        Start chat with docs, if the file is None, hold the current docs and continue chatting.
        :param file:
        :return:
        """
        if vector_db is not None:
            self.vec_db = vector_db
        # For Summarization at the start of the chat
        # Retrieval with current questions
        qa_with_docs = RetrievalQA.from_chain_type(llm=self.chat_model, chain_type='stuff',
                                                   retriever=self.vec_db.as_retriever())
        # Answer the question with the retrieved docs
        answers_with_docs = qa_with_docs.run(query)

        return self.generate_hold(answers_with_docs)

    def chat_depend_on_context(self,
                               query: str,
                               vector_db: Chroma = None
                               ) -> Generator:
        if vector_db is not None:
            self.vec_db = vector_db
        # For Summarization at the start of the chat
        # Semantic Search
        docs_retrieved = self.vec_db.similarity_search(query, k=4)
        res = self.llm_chain.predict(context=docs_retrieved, question=query)

        return res


class PDF_Center:
    def __init__(self,
                 pdf_proxy: WeResearch_PaperAss
                 ):
        self.proxy = pdf_proxy

    def load_pdf(self, file):
        accept_response = self.proxy.load_vec_db(file)
        if accept_response == '':
            return ''
        # Return the Value of Vector Database
        return accept_response

    def process(self, query: str, vector_db=None):
        if vector_db is not None:
            self.vec_db = vector_db

        return self.proxy.chat_model(query, self.vec_db)
