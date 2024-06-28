<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/capreolus/badge/?version=latest)](https://capreolus.readthedocs.io/?badge=latest)
[![PyPI version fury.io](https://badge.fury.io/py/capreolus.svg)](https://pypi.python.org/pypi/capreolus/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 


<h1>retrival25 👑</h1>

<i>One stop solution for all BM25 needs</i>
</div>

## Quick Start
Here is a simple example of how to use `retrival25`:

```python
from retrival25.rank import ranker

# Add you data here
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

# pass data to be preprocessed and indexed
obj = ranker(corpus=corpus, type="robertson", k=1.2, b=0.75)

# return top n retrived documents
obj.get_top_n(query=query[0], n=10)

```

## Variants 

You can use the following variants of BM25 

* Original implementation (`type="robertson"`)
* ATIRE (`type="atire_bm25"`)
* BM25L (`type="bm25_L"`)
* BM25+ (`type="bm25_plus+"`)
* BM25-adbt (`type="bm25_adbt"`)

By default ranker uses `type="robertson"` 