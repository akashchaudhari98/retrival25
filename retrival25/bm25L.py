import math
from collections import defaultdict
from retrival25.base import bm25


class bm25L(bm25):
    def __init__(
        self, document_corpus: list, b: float = 0.75, k: float = 1.2, epsilon=1
    ) -> None:
        super().__init__(document_corpus)
        self.b = b
        self.k = k
        self.term_doc_freq = {
            term: math.log((self.number_document + 1) / (freq + 0.5))
            for term, freq in self.term_doc_freq.items()
        }
        self.term_doc_freq = defaultdict(int)

    def idf(self, term) -> float:
        """Inverse document frequency"""

        return self.term_doc_freq[term]

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        tf = doc.count(term)

        ctd = tf / (1 - self.b + self.b * (len(doc) / self.avg_tok_doc))

        return (self.k + 1) * ctd / (self.k + ctd)
