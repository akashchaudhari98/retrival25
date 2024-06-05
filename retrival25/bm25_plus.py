import math
from retrival25.base import bm25


class bm25_plus(bm25):
    def __init__(
        self, document_corpus: list, b: float, k: float, epsilon: float
    ) -> None:
        super().__init__(document_corpus)
        self.b = b
        self.k = k
        self.epsilon = epsilon

    def idf(self, term) -> float:
        """Inverse document frequency"""

        dft = self.term_doc_freq[term]
        return math.log((self.number_document + 1) / (dft))

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        tf = doc.count(term)

        return (
            (self.k + 1)
            * tf
            / (self.k * ((1 - self.b) + self.b * ((len(doc) / self.avg_tok_doc))) + tf)
        ) + self.epsilon
