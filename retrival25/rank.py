from retrival25.robertson import robertson
from retrival25.atire_bm25 import atire_bm25
from retrival25.bm25L import bm25L
from retrival25.bm25_plus import bm25_plus
from retrival25.bm25_adpt import bm25_adbt

VARIANTS = {
    "robertson": robertson,
    "atire_bm25": atire_bm25,
    "bm25_L": bm25L,
    "bm25_plus": bm25_plus,
    "bm25_adbt": bm25_adbt,
}


class ranker:
    def __init__(
        self, corpus: list, k: float, b: float, epsilon: float, type: str = "robertson"
    ) -> None:
        self.bm25_obj = VARIANTS[type](document_corpus=corpus, b=b, k=k)

    def get_top_n(self, query: str, n=5):
        return self.bm25_obj.get_top_n(query, n)
