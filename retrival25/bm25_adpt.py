import math
from retrival25.base import bm25


class atire_bm25(bm25):
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

    def idf(self, term) -> float:
        """Inverse document frequency"""

        pass

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        
        pass
