import math
import numpy as np
from itertools import chain
from scipy.optimize import minimize
from retrival25.base import bm25
from collections import defaultdict


class bm25_adbt(bm25):
    def __init__(self, document_corpus: list, b: int, k: int) -> None:
        """_summary_

        Args:
            document_corpus (list): _description_
            k (int): _description_
            b (int, optional): _description_. Defaults to 0.
        """
        super().__init__(document_corpus)
        self.b = b
        self.k = k
        self.tokens = set(chain.from_iterable(self.corpus.values()))
        self._calc_ki()

    def _calc_ki(self):
        self.ki = defaultdict(lambda: 1)
        for token in self.tokens:
            ig = self._cal_info_gain(token)
            ig_1 = self._ig(token, 1)
            k_dash = minimize(self._k_dash, x0=ig_1, args=(ig_1, np.arange(len(ig))))
            self.ki[token] = k_dash.fun

    def _k_dash(self, ig, ig_1, t):
        return np.sum(((ig / ig_1) - ((self.k + 1) * t) / (self.k + t)) ** 2)

    def _cal_info_gain(self, term):
        previous_val, ig_val, t = 0, 0, 0
        ig = {}
        while True:
            ig_val = self._ig(term, t) + 0.001
            if previous_val >= ig_val:
                break
            else:
                ig[t] = ig_val
                previous_val = ig[t]
                t = t + 1
        return ig

    def _ig(self, term, t):
        return math.log(
            (self._df_t(term, t + 1) + 0.5) / (self._df_t(term, t) + 1)
        ) - math.log((self.term_doc_freq[term] + 0.5) / (self.number_document + 1))

    def _ctd(self, term, doc):
        return doc.count(term) / (1 - self.b + self.b * (len(doc) / self.avg_tok_doc))

    def _df_t(self, term, t):
        if t == 0:
            return self.number_document
        if t == 1:
            return self.term_doc_freq[term]
        if t > 1:
            return len(
                [
                    docs
                    for docs in self.corpus.values()
                    if term in docs and round(self._ctd(term, docs)) > t
                ]
            )

    def idf(self, term) -> float:
        """Inverse document frequency"""

        return self._ig(term, 1)

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        try:
            return ((self.ki[term] + 1) * doc.count(term)) / (
                self.ki[term] * ((1 - self.b) + self.b * ((len(doc) / self.avg_tok_doc)))
                + doc.count(term)
            )
        except Exception as ex:
            print(ex)
