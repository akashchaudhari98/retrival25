import math
from retrival25.base import bm25


class robertson(bm25):
    def __init__(
        self, document_corpus: list, b: float = 0.75, k: float = 1.2, epislon: float = 1
    ) -> None:
        super().__init__(document_corpus)
        self.b = b
        self.k = k
        self.term_doc_freq = {
            term: math.log((self.number_document - freq + 0.5) / (freq + 0.5))
            for term, freq in self.term_doc_freq.items()
        }

    def idf(self, term: str) -> float:
        """Inverse document frequency"""

        return self.term_doc_freq[term]

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        return doc.count(term) / (
            self.k * (1 - self.b + self.b * (len(doc) / self.avg_tok_doc))
            + doc.count(term)
        )
