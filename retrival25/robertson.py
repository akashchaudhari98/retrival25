import math
from retrival25.base import bm25

class robertson(bm25):
    def __init__(self, document_corpus: list, b:int, k:int) -> None:
        super().__init__(document_corpus)
        self.b = b
        self.k= k
 
    def idf(self, term) -> float:
        '''Inverse document frequency'''

        dft= len([doc for doc in self.corpus if term in doc])
        return math.log((self.number_document- dft + 0.5)/(dft + 0.5))

    def tf(self, term:str, doc:list) -> float:
        '''Term frequency'''
        return doc.count(term)/(self.k1*(1-self.b + self.b*(len(doc)/self.avg_tok_doc)) + doc.count(term))