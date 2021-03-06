import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        """
        logL:   log of the maximized value of the likelihood function for the estimated model
        logN:   log of the number of data points in x, the number of observations, or equivalently, the sample size
        p:      the number of free parameters to be estimated. If the estimated model is a linear regression, p is the number of
                regressors, including the intercept
        """
        lbicModel = (None, float("inf"))
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                p = n * (n-1) + (n-1) + 2 * self.X.shape[1] * n
                bic = (-2 * logL) + (p * np.log(self.X.shape[0]))
                lbicModel = (model, bic) if bic < lbicModel[1] else lbicModel
            except:
                continue

        return lbicModel[0]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    FFS... formula above looked wrong or easy to misunderstand... it should be...
    DIC = log(P(X(i)) - 1/(M-1) * SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        min_val = float("-inf")
        ldicModel = (None, float("-inf"))
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                sumAllLogL = sum(model.score(self.hwords[word][0], self.hwords[word][1]) for word in self.words)
                # print('ll', sumAllLogL)
                dic = logL - sumAllLogL/(len(self.words)-1)
                ldicModel = (model, dic) if dic > ldicModel[1] else ldicModel
            except:
                continue

        return ldicModel[0]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        means = []
        # fold split
        split_method = KFold()
        try:
            for nComp in self.n_components:
                model = self.base_model(nComp)
                folds = []
                for train, test in split_method.split(self.sequences):
                    X, length = combine_sequences(test, self.sequences)
                    folds.append(model.score(X, length))
                means.append(np.mean(fold_scores))
        except:
            pass

        states = self.n_components[np.argmax(means)] if means else self.n_constant
        return self.base_model(states)
