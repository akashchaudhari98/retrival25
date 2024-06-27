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
        self._precompute_ctd()
        self._calc_ki()

    def _precompute_ctd(self):
        self.term_ctd = defaultdict(lambda: defaultdict(int))
        for doc_id, doc in self.corpus.items():
            for term in doc:
                self.term_ctd[term][doc_id] = round(self._ctd(term, doc))

    def _calc_ki(self):
        self.ki = defaultdict(lambda: 1)
        for token in self.tokens:
            ig = self._cal_info_gain(token)

            k_dash = minimize(
                self._k_dash,
                x0=[self.k for _ in range(len(ig))],
                args=(
                    list(ig.values()),
                    [ig[1] for _ in range(len(ig))],
                    list(ig.keys()),
                ),
            )
            self.ki[token] = k_dash.fun

    def _k_dash(self, k, ig, ig_1, t):
        return np.sum(
            (
                np.array([ig[i] for i in t]) / ig_1
                - ((k + 1) * np.array(t)) / (k + np.array(t))
            )
            ** 2
        )

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
            return sum(1 for _, val in self.term_ctd[term].items() if val > 1)

    def idf(self, term) -> float:
        """Inverse document frequency"""

        return self._ig(term, 1)

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        return ((self.ki[term] + 1) * doc.count(term)) / (
            self.ki[term] * ((1 - self.b) + self.b * ((len(doc) / self.avg_tok_doc)))
            + doc.count(term)
        )
