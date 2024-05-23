import numpy as np
from abc import ABC, abstractmethod

class bm25:
    def __init__(self,document_corpus:list) -> None:
        self.corpus= [doc.split() for doc in document_corpus]
        self._initalise_variables__()
    
    def _initalise_variables__(self):
        self.number_document= len(self.corpus)
        self.avg_tok_doc= sum([len(doc) for doc in self.corpus])/self.number_document

    @abstractmethod
    def idf(self):
        pass
    
    @abstractmethod
    def tf(self):
        pass
    
    @abstractmethod
    def get_top_n(self):
        pass