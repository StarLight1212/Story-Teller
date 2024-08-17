from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
from utils.conversation import conv_templates, SeparatorStyle
from typing import Dict, Optional, List, Generator
from langchain.schema import Generation, LLMResult


def load_sentences2Ans(pth: str):
    df = pd.read_csv(pth, encoding='latin1')
    # Format: {index: [Question, Answer], }
    sentence_dict = {i: df['Answers'][i] for i in range(df.shape[0])}
    return sentence_dict, df['Question'].to_list()


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def tokenize_encode_norm_process(ref_sentences, ss_tokenizer, ss_model, device):
    # Tokenize sentences in demo-DB
    encoded_input = ss_tokenizer(ref_sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings in demo-DB
    with torch.no_grad():
        model_output = ss_model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def get_sentence_similarity_score(query_input, ref_sentence_embeddings, ss_tokenizer, ss_model, device):
    query_embedding = tokenize_encode_norm_process(query_input, ss_tokenizer, ss_model, device)
    cosine_scores = cos_sim(query_embedding, ref_sentence_embeddings)
    return cosine_scores


def mode_common_utils(cosine_scores) -> List:
    """Index and rank the similarity sentences."""
    pairs = []
    for i in range(len(cosine_scores[0])):
        pairs.append({'index': i, 'Score': cosine_scores[0][i]})
    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['Score'], reverse=True)
    return pairs


def debug_mode(query, sentences, cosine_scores, set_dict):
    pairs = mode_common_utils(cosine_scores)
    for pair in pairs[:10]:
        i = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f} Answer: {}".format(query, sentences[i], pair['Score'], set_dict[i]))
    return None


def test_mode(cosine_scores, threshold: float, set_dict):
    pairs = mode_common_utils(cosine_scores)
    best_score = pairs[0]['Score']
    ans_index = pairs[0]['index']
    # print(best_score)
    if best_score > threshold:
        # print('Find the Anchor Item!')
        return set_dict[ans_index]
    return None


class StoryLLM(LLM):
    """
    This code defines a StoryLLM class that extends the LLM class from langchain.llms.base module.
    It imports required libraries and contains several functions to tokenize, encode,
    and normalize sentences, calculate cosine similarity between input and reference sentences,
    and get the most similar sentence. The mode_common_utils function ranks the similarity of the
    sentences by their scores. If the score of the best-matching sentence is higher than a threshold,
    the test_mode function returns the corresponding answer. If not, it returns None.

    The StoryLLM class has the following attributes:
    - num_gpus: the number of GPUs to use for processing the model.
    - max_new_tokens: the maximum number of new tokens that can be added to the conversation.
    - temperature: the temperature for the softmax function used to sample the next token in the conversation.
    - top_p: the top_p value for the nucleus sampling.
    - device: the device (GPU or CPU) to use for processing the model.
    - tokenizer: an object to tokenize the input sentences.
    - model: an object to represent the model.
    - history_len: the maximum length of the history to keep track of.
    - conv_template: the conversation template to use.
    - conv: the conversation object.

    The WeResearch class has two methods:
    - __init__: initializes the class.
    - _call: processes the input and returns the output. If a history is provided,
    it is added to the conversation. The method then tokenizes, encodes, and normalizes the input sentences,
    calculates their cosine similarity with the reference sentences, and returns the answer with the highest
    similarity score if the score is higher than the threshold. Otherwise, it returns None.
    """
    num_gpus: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.80
    top_p: float = 0.35
    top_k: float = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device
    tokenizer: object = None
    model: object = None
    history_len: int = 10
    conv_template: str = "v1"
    conv = conv_templates[conv_template].copy()  # Conversation Start

    def __init__(self) -> None:
        super(WeResearch, self).__init__()

    @property
    def _llm_type(self) -> str:
        return "WeResearch"

    def __call__(self, inputs, history: Optional[List[Dict]] = None,
                 stream: bool = True, *args, **kwargs):
        """Call itself"""
        prompt = inputs
        if history is not None:
            return self._call(prompt, history, stream=stream)
        return self._call(prompt, stream=stream)

    def _call(self,
              prompt: str,
              history: Optional[List[Dict]] = None,
              stop: Optional[List[str]] = None,
              stream: bool = True) -> Generator or str:
        # History Loading
        if history is not None:
            for msg in history:
                self.conv.append_message(self.conv.roles[0], msg['usr_input'])
                self.conv.append_message(self.conv.roles[1], msg['weResearch'])

        # Chat Settings
        # Record user inputs history in current chat dialogue
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        # Construct prompt
        prompt = self.conv.get_prompt()
        params_ = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "stop": self.conv.sep if self.conv.sep_style == SeparatorStyle.SINGLE else self.conv.sep2,
        }
        # get response
        if not stream:
            return self.chat_normal(prompt, params_)
        return self.chat_stream(prompt, params_)

    def chat_normal(self,
                    prompt: str,
                    params) -> str:
        pre = 0
        answers = ''
        for outputs in self.generate_stream(self.tokenizer, self.model, params, self.device):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now - 1 > pre:
                answers += " ".join(outputs[pre:now]) + ' '
                pre = now
        answers += " ".join(outputs[pre:])
        # Record ResearchGPT Response Info
        self.conv.messages[-1][-1] = " ".join(outputs)

        return answers

    def chat_stream(self,
             prompt: str,
             params) -> Generator:
        pre = 0
        answers = ''
        for outputs in self.generate_stream(self.tokenizer, self.model, params, self.device):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now - 1 > pre:
                answers += " ".join(outputs[pre:now]) + ' '
                yield {"generated_text": answers}
                pre = now
        answers += " ".join(outputs[pre:])
        yield {"generated_text": answers}
        # Record ResearchGPT Response Info
        self.conv.messages[-1][-1] = " ".join(outputs)

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None,
            stream: bool = False,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, stream=stream)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def _agenerate(self,
                         prompts: List[str],
                         stop: Optional[List[str]] = None,
                         stream: bool = True
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        for prompt in prompts:
            text = await self._acall(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def async_generate(self,
                             prompts: List[str],
                             stop: Optional[List[str]] = None,
                             stream: bool = True
                             ):
        pass

    def load_model(self, model_name_or_path: str) -> None:
        """
        The model will start slowly at first,but
        will be accelerated from the second start time
        owing to it's automatic load from cache.
        :param model_name_or_path: model directory
        :return None
        """
        num_gpus = self.num_gpus

        # Model
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "23GiB" for i in range(num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        # Define Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        # Define Models
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            self.model.cuda()

    def top_k_top_p_filtering(self,
                              logits,
                              top_k: int = 0,
                              top_p: float = 0.0
                              ):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                :param filter_value:
                :param top_p: top p nucleic ranking methods
                :param logits:
                :param top_k:
        """
        if top_k > 0:
            filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
            logits[logits < filter[:, [-1]]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            filter = cumulative_probs > top_p
            # shift right by 1 since filter includes the first index that exceeds top_p
            filter[..., 1:] = filter[..., :-1].clone()
            filter[..., 0] = 0

            # convert to original indexing
            indices_to_remove = filter.scatter(1, sorted_indices, filter)
            logits[indices_to_remove] = float('-inf')

        return logits

    @torch.inference_mode()
    def generate_stream(self,
                        tokenizer,
                        model,
                        params,
                        device,
                        context_len: int = 2048,
                        stream_interval: int = 8):
        """
            Adapted from fastchat/serve/model_worker.py::generate_stream
        """

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", None))
        top_k = int(params.get("top_k", None))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device)
                out = model(input_ids=torch.as_tensor([[token]], device=device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                # get the last character logits to predict the next character, (B, C)
                last_token_logits = last_token_logits / temperature
                # Top k and Top p Sampling
                if top_p is not None and top_k is not None:
                    last_token_logits = self.top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)

                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break

        del past_key_values
