# Author: Chirag Jain
# Tom Kenter & Maarten de Rijke - Short Text Similarity with Word Embeddings

import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import scipy as sp
import tqdm
from gensim.models import FastText
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from sklearn.svm import SVC

eps = 1e-9


def _identity(text, **kwargs):
    return text


class GensimWordVectorizer(object):
    """
    Wrapper around gensim.models.base_any2vec.BaseAny2VecModel

    Provides non zero vector for unk words and other small properties
    """

    def __init__(self, model):
        self._model = model
        self._unk_vector = np.full((self._model.wv.vector_size,), eps, dtype=np.float32)

    def __call__(self, words):
        return [self._model.wv[word] if word in self else self._unk_vector for word in words]

    def __contains__(self, word):
        return word in self._model.wv

    def size(self):
        return self._model.wv.vector_size


class KenterSTS(object):
    """
    Train a short text similarity classifier

    training input
    pairs = [(pair1_text1, pair1_text2), ...,], labels = [1, 0, ...]

    """

    def __init__(self,
                 weights,
                 unk_weight,
                 vectorizer,
                 clf=SVC(kernel='rbf', class_weight='balanced', C=10e6, gamma=10e-4),
                 preprocessor=_identity,
                 preprocessor_kwargs=None,
                 bm25_k=1.2,
                 bm25_b=0.75,
                 merge_dim_features='diff',  # ['concat', 'diff', 'mul']
                 weighted_sailency_bins=(-np.inf, 0.16, 0.41, np.inf),
                 unweighted_mat_sailency_bins=(-np.inf, 0.46, 0.81, np.inf),
                 unweighted_max_sailency_bins=(-np.inf, 0.46, 0.81, np.inf),
                 dim_bins=(-np.inf, 0.0011, 0.011, 0.021, np.inf),
                 scale_features=False):
        """

        Args:
            weights (dict): mapping word to global importance score (say idf) {str: float, ... }
            unk_weight (float): weight to assign to words not in weights
            vectorizer (GensimWordVectorizer): GensimWordVectorizer object
            clf (sklearn classifier): Unfitted instance of sklearn classifier.
            preprocessor (function, optional): Any python function with a single argument
                                               to be called for preprocessing the text.
                                               It must return single value as unicode/str
                                               The tokens in the processed text are assumed to be delimited by space.
                                               Defaults to identity()
                                               Note: Avoid any heavy function. It will slow down training.
                                                     Instead preprocess the entire data beforehand
            preprocessor_kwargs (dict, optional): kwargs to pass to preprocessor. Defaults to None
            bm25_k (float, optional): Hyperparameter k, defaults to 1.2
            bm25_b (float, optional): Hyperparameter b, defaults to 0.75
            merge_dim_features (str, optional): how to merge sentence vectors before binning them,
                                                one of
                                                    'diff' - absolute difference in each dimension,
                                                    'concat' - concat the two vectors
                                                    'mul' - multiplication in each dimension
            weighted_sailency_bins (tuple, optional): bin ranges for binning weighted semantic similarity scores
                                                      between tokens of two sentences
                                                      (bin1_start_value, bin2_start_value, ..., max possible value)
                                                      See 3.1.1 in paper
                                                      Defaults to (-np.inf, 0.16, 0.41, np.inf)
            unweighted_mat_sailency_bins (tuple, optional): bin ranges for binning semantic similarity matrix
                                                            scores between two sentences
                                                            (bin1_start_value, ..., max possible value)
                                                            See 3.1.2 in paper
                                                            Defaults to (-np.inf, 0.46, 0.81, np.inf)
            unweighted_max_sailency_bins (tuple, optional): bin ranges for binning max semantic similarity scores along
                                                            each row in similarity matrix
                                                            (bin1_start_value, ..., max possible value)
                                                            See 3.1.2 in paper
                                                            Defaults to (-np.inf, 0.46, 0.81, np.inf)
            dim_bins (tuple, optional): bin ranges for binning dimensions values of merged sentence vector.
                                        (bin1_start_value, bin2_start_value, bin3_start_value, ..., max possible value)
                                        See 3.2.2 in paper
                                        Defaults to (-np.inf, 0.0011, 0.011, 0.021, np.inf)
            scale_features (bool, optional): scale each feature to zero mean and unit variance. Defaults to False
        """
        self.weights = weights
        self.unk_weight = unk_weight
        self.avg_doc_length = 5
        self.bm25_k = bm25_k
        self.bm25_b = bm25_b
        self._merge_dim_features = merge_dim_features
        self.weighted_sailency_bins = weighted_sailency_bins
        self.unweighted_mat_sailency_bins = unweighted_mat_sailency_bins
        self.unweighted_max_sailency_bins = unweighted_max_sailency_bins
        self.dim_bins = dim_bins
        self._vectorizer = vectorizer
        self._preprocessor = preprocessor
        self._preprocessor_kwargs = preprocessor_kwargs or {}
        self._scale_features = scale_features
        self._clf = clf
        self._clf_save_path = None
        self._fitted = False

    def __getstate__(self):
        return (
            self.weights,
            self.unk_weight,
            self.avg_doc_length,
            self.bm25_k,
            self.bm25_b,
            self._merge_dim_features,
            self.weighted_sailency_bins,
            self.unweighted_mat_sailency_bins,
            self.unweighted_max_sailency_bins,
            self.dim_bins,
            self._scale_features,
            self._clf_save_path,
            self._fitted
        )

    def __setstate__(self, state):
        (
            self.weights,
            self.unk_weight,
            self.avg_doc_length,
            self.bm25_k,
            self.bm25_b,
            self._merge_dim_features,
            self.weighted_sailency_bins,
            self.unweighted_mat_sailency_bins,
            self.unweighted_max_sailency_bins,
            self.dim_bins,
            self._scale_features,
            self._clf_save_path,
            self._fitted
        ) = state

        self._clf = joblib.load(self._clf_save_path)
        self._vectorizer = None
        self._preprocessor = _identity
        self._preprocessor_kwargs = {}
        print('Please set vectorizer and preprocessor function by calling set_vectorizer and set_preprocessor'
              'if you set any custom ones before saving!')

    def save(self, fname):
        self._clf_save_path = fname + '.sklearn'
        joblib.dump(self._clf, self._clf_save_path)
        pickle.dump(self, open(fname, 'wb'), 2)
        # Since vectorizer can be giant blobs, we will avoid saving them
        # Same for preprocessor as they can be anywhere in external scope
        print('Note: vectorizer and preprocessor will not be saved!'
              '\nPlease ensure you can set them separately during load\n')

    @staticmethod
    def load(fname):
        """
        Args:
            fname (str): path to saved model

        Returns:
            KenterSTS: instance without vectorizer and preprocessor set
        """
        return pickle.load(open(fname, 'rb'))

    def set_vectorizer(self, vectorizer):
        self._vectorizer = vectorizer

    def set_preprocessor(self, preprocessor, preprocessor_kwargs):
        self._preprocessor = preprocessor
        self._preprocessor_kwargs = preprocessor_kwargs

    def _get_features(self, text1, text2):
        def _pairwise_sim(_token, _doc):
            return [self.word_sim(_token, other_token) for other_token in _doc]

        doc1 = text1.split()
        doc2 = text2.split()
        if len(doc1) > len(doc2):
            doc1, doc2 = doc2, doc1

        weighted_scores = []
        sim_scores = []
        maxsim_scores = []

        for token in doc1:
            cached = _pairwise_sim(token, doc2)
            sim_scores.extend(cached)

            max_cached = max(cached)
            maxsim_scores.append(max_cached)

            num = self.weights.get(token, self.unk_weight) * max_cached * (self.bm25_k + 1)
            den = max_cached + self.bm25_k * (1 - self.bm25_b + self.bm25_b * (len(doc1) / float(self.avg_doc_length)))
            weighted_scores.append(num / den)

        feature_set1 = self._count_bins(weighted_scores, bins=self.weighted_sailency_bins, normalize=True)
        feature_set2 = self._count_bins(sim_scores, bins=self.unweighted_mat_sailency_bins, normalize=True)
        feature_set3 = self._count_bins(maxsim_scores, bins=self.unweighted_max_sailency_bins, normalize=True)

        vec1 = self.get_sentence_vector(text1)
        vec2 = self.get_sentence_vector(text2)
        feature_set4 = np.array([sp.spatial.distance.euclidean(vec1, vec2),
                                 1.0 - sp.spatial.distance.cosine(vec1, vec2)], dtype=np.float32)

        if self._merge_dim_features == 'mul':
            vec = np.multiply(vec1, vec2)
        elif self._merge_dim_features == 'diff':
            vec = np.abs(vec1 - vec2)
        else:
            vec = np.concatenate((vec1, vec2))

        feature_set5 = self._count_bins(list(vec), bins=self.dim_bins, normalize=True)
        features = np.concatenate([feature_set1, feature_set2, feature_set3, feature_set4, feature_set5])

        return features

    def word_sim(self, word1, word2):
        vec1, vec2 = self._vectorizer([word1, word2])
        return 1.0 - sp.spatial.distance.cosine(vec1, vec2)

    def get_sentence_vector(self, sentence, normalize=False):
        tokens = sentence.split()
        vec = np.array(self._vectorizer(tokens))
        vec = np.mean(vec, axis=0)
        vec = np.ravel(vec)

        if normalize:
            vec /= (np.linalg.norm(vec, 2) + eps)

        if not np.all(np.isfinite(vec)):
            print("Inf:", sentence)

        return vec

    def _count_bins(self, a, bins, normalize):
        d = np.bincount(np.digitize(a, bins) - 1, minlength=len(bins) - 1).astype(np.float32)
        if normalize:
            d /= float(len(a))
        return d

    def fit(self, pairs, labels):
        if self._fitted:
            return self

        self._clf.fit(self.transform(pairs), labels)
        self._fitted = True
        return self

    def transform(self, pairs):
        vectors = []
        _pairs = []

        if not self._fitted:
            self.avg_doc_length = 0

        for text1, text2 in pairs:
            text1 = self._preprocessor(text1, **self._preprocessor_kwargs)
            text2 = self._preprocessor(text2, **self._preprocessor_kwargs)
            _pairs.append((text1, text2))
            if not self._fitted:
                self.avg_doc_length += len(text1.split())
                self.avg_doc_length += len(text2.split())

        if not self._fitted:
            self.avg_doc_length /= (2.0 * len(pairs))

        for text1, text2 in tqdm.tqdm(_pairs):
            v = self._get_features(text1, text2)
            vectors.append([v])

        X = np.concatenate(vectors, axis=0)
        if self._scale_features:
            X = scale(X)

        return X

    def fit_transform(self, pairs, labels):
        X = self.transform(pairs)

        if self._fitted:
            return X

        self._clf.fit(X, labels)
        self._fitted = True
        return X

    def predict(self, pairs):
        return self._clf.predict(self.transform(pairs))

    def predict_proba(self, pairs):
        return self._clf.predict_proba(self.transform(pairs))


if __name__ == '__main__':
    sample_data = [(u'hello', u'hi'), (u'i like this', u'i hate it')]
    sample_labels = [1, 0]
    sample_weights = {u'hello': 1, u'hi': 1, u'i': 0.1, u'like': 1, u'this': 0.5, u'hate': 0.9, u'it': 0.5}
    _docs = []
    for a, b in sample_data:
        _docs.append(a.split())
        _docs.append(b.split())
    sample_unk_weight = 0.5
    sample_vectorizer = GensimWordVectorizer(FastText(_docs, min_count=1))
    model = KenterSTS(sample_weights, sample_unk_weight, vectorizer=sample_vectorizer)
    model.fit(sample_data, sample_labels)
    model.save('test_save')
    model = KenterSTS.load('test_save')
    model.set_vectorizer(sample_vectorizer)
    print("Test Prediction:", model.predict([(u'hello', u'hi')]))
    os.remove('test_save')
    os.remove('test_save.sklearn')
