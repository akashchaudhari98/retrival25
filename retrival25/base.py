from abc import abstractmethod
from tok import word_tokenize
from collections import defaultdict
import copy
import math


class bm25:

    def __init__(self, document_corpus: list) -> None:
        self.corpus = dict(enumerate(document_corpus))
        self.corpus_copy = copy.deepcopy(self.corpus)
        self.corpus = {i: word_tokenize(doc) for i, doc in self.corpus.items()}
        self.__initalise_variables__()

    def __initalise_variables__(self):
        self.number_document = max(self.corpus.keys()) + 1

        len_of_doc = [len(doc) for doc in self.corpus.values()]
        self.avg_tok_doc = sum(len_of_doc) / self.number_document

        self.term_doc_freq = defaultdict(int)
        for doc in self.corpus.values():
            # term_freq = Counter(doc)
            for term in set(doc):
                self.term_doc_freq[term] += 1
        self.term_doc_freq = {
            term: math.log(self.number_document / freq)
            for term, freq in self.term_doc_freq.items()
        }
        self.term_doc_freq = defaultdict(lambda: 1, self.term_doc_freq)

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
        toknised_query = word_tokenize(query)
        scores = {
            id: [doc, self.score(toknised_query, doc)]
            for id, doc in self.corpus.items()
        }
        sorted_scores = [
            [self.corpus_copy[id], v[1]]
            for id, v in sorted(
                scores.items(), key=lambda item: item[1][1], reverse=True
            )[:n]
        ]
        return sorted_scores
