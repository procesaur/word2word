# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from itertools import chain, product
import operator
from tqdm import tqdm
from transformers import AutoTokenizer


def load_hf_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def load_word_tokenizer(lang):
    if lang == "ko":
        from konlpy.tag import Mecab
        tokenizer = Mecab()
    elif lang == "ja":
        import Mykytea
        opt = "-model jp-0.4.7-1.mod"
        tokenizer = Mykytea.Mykytea(opt)
    elif lang == "zh_cn":
        import Mykytea
        opt = "-model ctb-0.4.0-1.mod"
        tokenizer = Mykytea.Mykytea(opt)
    elif lang == "zh_tw":
        import jieba
        tokenizer = jieba
    elif lang == "vi":
        from pyvi import ViTokenizer
        tokenizer = ViTokenizer
    elif lang == "th":
        from pythainlp.tokenize import word_tokenize
        tokenizer = word_tokenize
    elif lang == "ar":
        import pyarabic.araby as araby
        tokenizer = araby
    else:
        from nltk.tokenize import ToktokTokenizer
        tokenizer = ToktokTokenizer()
    return tokenizer


def get_vocab(dataset, column):
    word2idx, idx2word, idx2cnt = dict(), dict(), dict()
    X = [ex[column] for ex in dataset]
    word2cnt = Counter(list(chain.from_iterable(X))).most_common()
    word2cnt.sort(key=operator.itemgetter(1, 0), reverse=True)
    for idx, (word, cnt) in enumerate(tqdm(word2cnt)):
        word2idx[word] = idx
        idx2word[idx] = word
        idx2cnt[idx] = cnt

    return word2idx, idx2word, idx2cnt


def update_dicts(dataset, lang1, lang2, vocab1, vocab2, cutoff, n_lines, save_pmi):
    """Get monolingual and cross-lingual count dictionaries.

    'cutoff' determines how many collocates are considered in each language.
    """

    def u2_iter(t1, t2, same_ignore=False, cut_t2=None):
        for _ in product(t1, t2):
            if (not same_ignore or _[0] != _[1]) and (not cut_t2 or _[1] < cut_t2):
                yield _

    def build_ddi():
        return defaultdict(lambda: defaultdict(int))

    x_x_dict = build_ddi()
    y_y_dict = build_ddi()
    x_y_dict = build_ddi()
    y_x_dict = build_ddi()
    seqlens1 = []
    seqlens2 = []

    for ex in tqdm(dataset, total=n_lines):

        if save_pmi:
            seqlens1.append(len(ex[lang1]))
            seqlens2.append(len(ex[lang2]))

        xs = [vocab1[wx] for wx in ex[lang1] if wx in vocab1]
        ys = [vocab2[wy] for wy in ex[lang2] if wy in vocab2]

        for xx1, xx2 in u2_iter(xs, xs, same_ignore=True, cut_t2=cutoff):
            x_x_dict[xx1][xx2] += 1
        for yy1, yy2 in u2_iter(ys, ys, same_ignore=True, cut_t2=cutoff):
            y_y_dict[yy1][yy2] += 1
        for xx, yy in u2_iter(xs, ys, same_ignore=False):
            x_y_dict[xx][yy] += 1
            y_x_dict[yy][xx] = x_y_dict[xx][yy]

    # convert to ordinary dicts for pickling
    def ddi2dict(ddi):
        return {k: dict(v) for k, v in ddi.items()}

    return tuple(
        list(ddi2dict(ddi) for ddi in [x_x_dict, y_y_dict, x_y_dict, y_x_dict])
        + [seqlens1, seqlens2]
    )
