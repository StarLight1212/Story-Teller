import pickle
from utils.common_utils import DFAFilter


def load_dict_txt(pth: str):
    sens_words = []
    f = open(pth, 'r', encoding='utf-8')
    for line in f.readlines():
        sens_words.append(line[:-1])
    return sens_words


tmp_filter = DFAFilter()

zh_pth = "./sensitive_txt/chinese_dictionary.txt"
en_pth = "./sensitive_txt/english_dictionary.txt"

zh_sen_words = load_dict_txt(zh_pth)
en_sen_words = load_dict_txt(en_pth)

# Add all sensitive lst together
aggregated_lst = zh_sen_words + en_sen_words

for item in aggregated_lst:
    tmp_filter.add(item)

pickle.dump(tmp_filter.keyword_chains, open("sensitive_corups.pkl", "wb"))
