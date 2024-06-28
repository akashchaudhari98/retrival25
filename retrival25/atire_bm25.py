import math
from retrival25.base import bm25


class atire_bm25(bm25):
    def __init__(
        self, document_corpus: list, b: float = 0.75, k: float = 1.2, epsilon: float = 1
    ) -> None:
        """_summary_

        Args:
            document_corpus (list): _description_
            k (int): _description_
            b (int, optional): _description_. Defaults to 0.
        """
        super().__init__(document_corpus)
        self.b = b
        self.k = k
        self.term_doc_freq = {
            term: math.log(self.number_document / freq)
            for term, freq in self.term_doc_freq.items()
        }

    def idf(self, term) -> float:
        """Inverse document frequency"""

        return self.term_doc_freq[term]

    def tf(self, term: str, doc: list) -> float:
        """Term frequency"""
        tf = doc.count(term)
        return (
            (self.k + 1)
            * tf
            / (self.k * (1 - self.b + self.b * (len(doc) / self.avg_tok_doc)) + tf)
        )
