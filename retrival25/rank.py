from robertson import robertson

VARIANTS= {
    "robertson": robertson
}
class ranker:
    def __init__(self, corpus: list, type:str) -> None:
        bm25_obj= VARIANTS[type]
        pass
