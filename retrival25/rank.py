from retrival25.robertson import robertson
from retrival25.atire_bm25 import atire_bm25
from retrival25.bm25L import bm25L

VARIANTS = {"robertson": robertson, "atire_bm25": atire_bm25, "bm25_L": bm25L}


class ranker:
    def __init__(
        self, corpus: list, k: float, b: float, epsilon: float,  type: str = "robertson"
    ) -> None:
        self.bm25_obj = VARIANTS[type](document_corpus=corpus, b=b, k=k)

    def get_top_n(self, query: str, n=5):
        return self.bm25_obj.get_top_n(query, n)
