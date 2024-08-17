import tornado.web
import json
import pickle
from utils.settings import *
from vectorData.chroma import Chroma
from utils.common_utils import DFAFilter
from llmCore.StoryCore import StoryLLM
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from researchProgram.weResearchOrigin import WeResearch_Model
from researchProgram.SemanticSearch_exp import SemanticsFam_Expr
from researchProgram.PaperAssistant import WeResearch_PaperAss


# ------------Settings ---------
# Wors Ban Path Define Region
wd_ban_pth = "./dataRestore/words_ban/sensitive_corups.pkl"
# Load sensitive information
sens_corups = pickle.load(open(wd_ban_pth, "rb"))
# Loading sensitive words proxy
filter_proxy = DFAFilter()
# Input the sens_corpus into the DFA filter
filter_proxy.keyword_chains = sens_corups

# Setting for Embeddings Layer
# embeddings_layer = HuggingFaceInstructEmbeddings(model_name=embeddings_path)
embeddings_layer = HuggingFaceEmbeddings()

# Load Vector Database
vector_db = Chroma(collection_name=COLLECTION_NAME,
                   embedding_function=embeddings_layer,
                   persist_directory=PERSISTENCE_DIR)

# $1. StoryLLM Model Initiate & Loading
storyLLM = StoryLLM()
storyLLM.load_model(chat_weights)

# $2. Sentence Similarity Discrimination Model
ss_model = AutoModel.from_pretrained(ss_weights)
ss_tokenizer = AutoTokenizer.from_pretrained(ss_weights)

# $3.1 Original Chatbot for Normal Chatting
weResearch_chat = WeResearch_Model(storyLLM, ss_model, ss_tokenizer,
                                   pattern_pth=pattern_path)

# $3.2 Experiment Assistant
exp_assistant = SemanticsFam_Expr(storyLLM, ss_model, ss_tokenizer,
                               pattern_path, embeddings_model=embeddings_layer,
                               persist_path='./dataRestore/chroma_storage_exp')

# $3.3 Paper Reads' Assistant
docs_assistant = WeResearch_PaperAss(storyLLM, ss_model, ss_tokenizer,
                               pattern_path, embeddings_model=embeddings_layer)

# answers_with_docs = docs_assistant.chat_with_docs(query)


class WeResearch(tornado.web.RequestHandler):
    def post(self):
        # 从POST请求的body中获取输入的JSON数据
        request_data = json.loads(self.request.body)

        # Load user input data and history data from JSON
        input_text = request_data.get('input_prompt')
        history = request_data.get('history')
        # Filter for user query
        # query_text = filter_proxy.filter(message=input_text)
        # is_sensitive = 1 if '*' in query_text else 0
        # Large Language Model Prediction
        predicted_text = weResearch_chat.process(input_text, history)
        # 设置HTTP响应头信息，指定数据格式为JSON
        self.set_header("Content-Type", "application/json")

        # 使用Tornado的异步HTTP响应对象，将生成器对象返回
        index_len = 0
        for chunk in predicted_text:
            filtered_chunk = filter_proxy.filter(message=chunk['generated_text'])
            chunk = {'generated_text': filtered_chunk[index_len:]}
            index_len = len(filtered_chunk)
            self.write(json.dumps(chunk))
            self.flush()


class ExperimentAssistant(tornado.web.RequestHandler):
    def post(self):
        # 从POST请求的body中获取输入的JSON数据
        request_data = json.loads(self.request.body)

        # Load user input data and history data from JSON
        input_text = request_data.get('input_prompt')
        history = request_data.get('history')
        # Large Language Model Prediction
        predicted_text = exp_assistant.experiment_assistant(input_text, history)
        # 设置HTTP响应头信息，指定数据格式为JSON
        self.set_header("Content-Type", "application/json")

        # 使用Tornado的异步HTTP响应对象，将生成器对象返回
        index_len = 0
        for chunk in predicted_text:
            filtered_chunk = filter_proxy.filter(message=chunk['generated_text'])
            chunk = {'generated_text': filtered_chunk[index_len:]}
            index_len = len(filtered_chunk)
            self.write(json.dumps(chunk))
            self.flush()


class PaperAssistant(tornado.web.RequestHandler):
    def post(self):
        # 从POST请求的body中获取输入的JSON数据
        # Load user input data and history data from JSON
        input_text = self.get_argument('input_prompt', None)
        has_token = self.get_argument('hash_token', None)
        if input_text is None:
            # Get Files
            file = self.request.files['file'][0]
            user_vec_db = docs_assistant.load_vec_db(file, vector_db, has_token)
            predicted_text = docs_assistant.chat_with_docs('Please summarize the current document.', user_vec_db)
        else:
            # Large Language Model Prediction
            predicted_text = docs_assistant.chat_with_docs(input_text)
        # 设置HTTP响应头信息，指定数据格式为JSON
        self.set_header("Content-Type", "application/json")

        # 使用Tornado的异步HTTP响应对象，将生成器对象返回
        index_len = 0
        for chunk in predicted_text:
            if input_text is not None:
                filtered_chunk = filter_proxy.filter(message=chunk['generated_text'])
                chunk = {'generated_text': filtered_chunk[index_len:]}
                index_len = len(filtered_chunk)
            self.write(json.dumps(chunk))
            self.flush()


if __name__ == "__main__":
    # 创建Tornado应用程序
    app = tornado.web.Application([
        (r"/weresearch", WeResearch),
        (r"/chat_with_protocol", ExperimentAssistant),
        (r"/chat_with_paper", PaperAssistant)
    ])
    # 启动Tornado服务
    app.listen(7860)
    tornado.ioloop.IOLoop.current().start()
