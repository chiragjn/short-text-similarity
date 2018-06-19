### Short Text Similarity with word embedding vectors

----

Quick Implementation of STS model as described in [Tom Kenter & Maarten de Rijke - Short Text Similarity with Word Embeddings](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/kenter-short-2015.pdf)

> Caution: Few assumptions were made as they were a bit unclear in the paper

`KenterSTS` module in `sts.py` has short python doc referring to hyperparams in paper

See `main` part of `sts.py` for sample usage

```python
from sts import *

sample_data = [(u'hello', u'hi'), (u'i like this', u'i hate it')]
sample_labels = [1, 0]
sample_weights = {u'hello': 1, u'hi': 1, u'i': 0.1, u'like': 1, u'this': 0.5, u'hate':0.9, u'it': 0.5}
_docs = []
for a, b in sample_data:
    _docs.append(a.split())
    _docs.append(b.split())
sample_unk_weight = 0.5
sample_vectorizer = GensimWordVectorizer(FastText(_docs, min_count=1)) # plug any gensim word vector model here
model = KenterSTS(sample_weights, sample_unk_weight, vectorizer=sample_vectorizer)
model.fit(sample_data, sample_labels)
model.save('test_save')
model = KenterSTS.load('test_save')
model.set_vectorizer(sample_vectorizer)
print("Test Prediction:", model.predict([(u'hello', u'hi')]))
os.remove('test_save')
os.remove('test_save.sklearn')
```



### Additional Caution

-----

This module works with dense vectors only!

Considering the default params generate about 15 feature per pair this shouldn't be a problem for moderately large datasets

If you are increasing bins and have a huge dataset, consider modifying the code to work with sparse matrices.

