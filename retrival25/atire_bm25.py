from retrival25.base import bm25

class atire_bm25(bm25):
    def __init__(self,document_corpus: list, b:int, k1:int) -> None:
        super().__init__(document_corpus)
        self.b = b
        self.k1= k1
    
    def idf(self, term) -> float:
        '''Inverse document frequency'''
        dft= len([doc for doc in self.corpus if term in doc])

        return self.number_document/dft
        
    def tf(self, term:str, doc:list) -> float:
        '''Term frequency'''
        tf= doc.count(term)
        return (self.k1 + 1)*tf/(self.k1*(1 - self.b + self.b*(len(doc)/self.avg_tok_doc)) + tf)
    
    def score(self, query:list, doc:list) -> float:
        return sum([self.idf(term)*self.tf(term, doc) for term in query])

    def get_top_n(self, query:str, n= 5)-> dict:
        '''Retrive top n document from corpus'''    
        toknised_query= query.split()
        scores= {doc: self.score(toknised_query, doc) for doc in self.corpus}
        sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])[:n]}
        return sorted_scores