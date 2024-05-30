from abc import abstractmethod
import time
from collections import defaultdict, Counter


class bm25:

    def __init__(self, document_corpus: list) -> None:
        self.corpus = [doc.split() for doc in document_corpus]
        self._initalise_variables__()

    def _initalise_variables__(self):
        self.number_document = len(self.corpus)

        len_of_doc = [len(doc) for doc in self.corpus]
        self.avg_tok_doc = sum(len_of_doc) / self.number_document

        self.term_doc_freq = defaultdict(int)
        self.doc_term_freq = []
        for doc in self.corpus:
            term_freq = Counter(doc)
            self.doc_term_freq.append(term_freq)
            for term in term_freq:
                self.term_doc_freq[term] += 1

    @abstractmethod
    def idf(self):
        pass

    @abstractmethod
    def tf(self):
        pass

    def score(self, query: list, doc: list) -> float:
        return sum([self.idf(term) * self.tf(term, doc) for term in query])

    def get_top_n(self, query: str, n=5) -> dict:
        """Retrive top n document from corpus"""
        toknised_query = query.split()
        scores = [[doc, self.score(toknised_query, doc)] for doc in self.corpus]
        sorted_scores = [
            [k, v] for k, v in sorted(scores, key=lambda item: item[1])[:n]
        ]
        return sorted_scores
