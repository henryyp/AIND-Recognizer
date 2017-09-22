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
        lbicModel = (None, float("inf"))
        """
        logL:   log of the maximized value of the likelihood function for the estimated model
        logN:   log of the number of data points in x, the number of observations, or equivalently, the sample size
        p:      the number of free parameters to be estimated. If the estimated model is a linear regression, p is the number of
                regressors, including the intercept
        """
        try:
            for n in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                p = n * (n-1) + (n-1) + 2 * self.X.shape[1] * n
                bic = (-2 * logL) + (p * np.log(self.X.shape[0])
                lbicModel = (model, bic) if bic < lbicModel[1] else lbicModel
        except:
            pass
        return lbicModel[0]


class SelectorDIC(ModelSelector):

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        min_val = float("-inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                total_other_logL = 0
                for word in self.words:
                    other_x, other_lengths = self.hwords[word]
                    total_other_logL += model.score(other_x, other_lengths)
                avg_logL = total_other_logL/(len(self.words)-1)
                dic_score = logL - avg_logL
                if dic_score > min_val:
                    min_val = dic_score
                    best_model = model
            except:
                continue
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds'''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        mean_scores = []
        # Save reference to 'KFold' in variable as shown in notebook
        split_method = KFold()
        try:
            for n_component in self.n_components:
                model = self.base_model(n_component)
                # Fold and calculate model mean scores
                fold_scores = []
                for _, test_idx in split_method.split(self.sequences):
                    # Get test sequences
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    # Record each model score
                    fold_scores.append(model.score(test_X, test_length))

                # Compute mean of all fold scores
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        states = self.n_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(states)
