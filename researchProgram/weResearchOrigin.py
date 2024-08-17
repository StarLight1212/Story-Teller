from llmCore.basic import ResearchBase
import torch
from llmCore.weResearchCore import WeResearch, load_sentences2Ans, \
    tokenize_encode_norm_process, test_mode, get_sentence_similarity_score
from typing import Dict, Optional, List, Generator


class WeResearch_Model(ResearchBase):
    # Threshold for selection the Pattern and AI Chat Mode
    num_gpus = 1
    threshold = 0.8
    load_8bit: bool = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device

    def __init__(self,
                 chat_model: WeResearch,
                 ss_model,
                 ss_tokenizer,
                 pattern_pth: str,
                 stream: bool = True  # test_mode, debug_mode
                 ) -> None:
        super(WeResearch_Model, self).__init__()
        self.stream = stream
        # Loading Chat Model
        self.chat_model = chat_model
        # Similarity sentences loading
        self.set_dict, self.ref_sentences = load_sentences2Ans(pattern_pth)
        # Load sentence similarity detection model from the Path
        self.ss_tokenizer = ss_tokenizer
        self.ss_model = ss_model.to(self.device)

        # Get Sentence Embedded Vector
        self.ref_sentences_embedded = tokenize_encode_norm_process(self.ref_sentences, self.ss_tokenizer, self.ss_model,
                                                                   self.device)

    def pattern_recom(self, pat_ans: str) -> Generator:
        yield {"generated_text": pat_ans}

    def process(self, usr_query: str,
                history: Optional[List[Dict]] = None) -> Generator or WeResearch:
        # Pattern Recommendation
        # Calculate Sentence Similarity Score, cosine theta
        cosine_scores = get_sentence_similarity_score(usr_query, self.ref_sentences_embedded, self.ss_tokenizer,
                                                      self.ss_model, self.device)
        pattern_answer = test_mode(cosine_scores, self.threshold, self.set_dict)

        if pattern_answer is not None:
            return self.pattern_recom(pattern_answer)

        # Chat Settings
        if history is not None:
            return self.chat_model(usr_query, history)

        return self.chat_model(usr_query)