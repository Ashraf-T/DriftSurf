import numpy
from scipy.special import expit
import logging
from statistics import mean
import collections

class Opt:
    """
    A class used to determine the optimization problem solver: SGD or STRSAGA

    ...

    Attributes
    ----------
    SGD : str
        Stochastic Gradient Descent
    STRSAGA : str
        A variance reduced version of SGD
    """

    SGD = 'sgd'
    STRSAGA = 'strsaga'

class Model:
    """
    A class used to define the model

    Methods
    -------
    update_step(training_point, step_size, mu)
        based on the optimization problem, solver calls the proper updating method

    sgd_step(training_point, step_size, mu)
        updates the model using sgd method in the single model setting

    saga_step(training_point, step_size, mu)
        updates the model using saga method in the single model setting

    loss(data)
        computes the logistic loss for the given dataset

    reg_loss(data, mu)
        computes the L2-regularized logistic loss for the given dataset
    """

    def update_step(self, training_point, step_size, mu):
        """ based on the optimization problem solver calls the proper updating method

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        """

        if self.opt == Opt.SGD:
            self.sgd_step(training_point, step_size, mu)
            return 1
        elif self.opt == Opt.STRSAGA:
            self.strsaga_step(training_point, step_size, mu)
            return 1

    def sgd_step(self, training_point, step_size, mu):
        """updates the model using sgd method

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const

        :raise NotImplementedError
            This abstract method will be override in the derived classes

        """

        raise NotImplementedError()

    def strsaga_step(self, training_point, step_size, mu):
        """updates the model using saga method in the single model setting

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const

        :raise NotImplementedError
            This abstract method will be override in the derived classes
        """

        raise NotImplementedError()

    def loss(self, data):
        """computes the logistic loss for the given data

        :param data: list of tuples: [(int, {int:float}, int)]
            A list of data points where each data point is in the form of (i, x, y) where i is its index,
            x is a dictionary of the features (key, value) pairs, and y is the label which can be either 1 or -1
        :raise NotImplementedError
            This abstract method will be override in the derived classes
        """

        raise NotImplementedError()

    def reg_loss(self, data, mu):
        """computes the L2-regularized logistic loss for the given data

        :param data: list of tuples : [(int, {int:float}, int)]
            A list of data points where each data point is in the form of (i, x, y) where i is its index,
            x is a dictionary of the features (key, value) pairs, and y is the label which can be either 1 or -1
        :param mu: float
            L2-regularization const

        :raise NotImplementedError
            This abstract method will be override in the derived classes
        """

        raise NotImplementedError()

class LogisticRegression_expert(Model):
    """
    A class to define a single expert which is a children of Model
    ...

    Attributes
    ----------
    param : ndarray
        model parameters for the expert
    weight: float
        weight of the expert (default is 1)
    opt : str
        optimization problem solver, either SGD or STRSAGA
    T_pointers : tuples : (int, int)
        pointers to the start and end of the effective sample set of the expert (default is (0,0))
    table: dict : {int: float}
        a dictionary mapping the last computed gradient with respect to a data point
    table_sum: ndarray
        an average over all the most recent computed gradients with respect to all data points seen so far
    perf: (float, float, float)
        performance of the expert (current, previous, best)
    """

    def __init__(self, init_param, opt, buffer_pointer=0, weight=1, r=4):
        """

        :param init_param:
                    parameter to initialize the model's parameter
        :param opt:
                    optimization problem solver, either SGD or SAGA
        :param buffer_pointer:
                    the current pointer in the buffer which indicates the start of the effective sample set (default is 0)
        :param weight:
                    weight of the expert (if used in ensemble of learners, default value is 1)
        """

        self.param = init_param
        self.weight = weight
        self.opt = opt

        self.T_pointers = (buffer_pointer, buffer_pointer)

        self.table = {}
        self.table_sum = numpy.zeros(self.param.shape)

        self.perf = (None, None, None) # current, previous, best observed, avg over last r time steps
        self.sliding_window = []
        self.r = r

    def dot_product(self, x):
        """ computes the dot product of input x's features and the model parameter

        :param x: dict : {int:float}
            a dictionary of input's feature in form of (key, value)
        :return: float
            returns the dot product of input x's features and the model parameter
        """

        return sum(self.param[k]*v for (k,v) in x.items())

    def sgd_step(self, training_point, step_size, mu):
        """ updates the model using sgd method

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        """

        (i, x, y) = training_point
        p = 1. / (1 + numpy.exp(y * self.dot_product(x)))

        self.param[:] = (1 - step_size * mu) * self.param
        for (k, v) in x.items():
            self.param[k] -= step_size * (-1 * p * y * v)

    def sgd_step_biased(self, training_point, step_size, mu, wp):
        """ updates the model using sgd method

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        """

        (i, x, y) = training_point
        p = 1. / (1 + numpy.exp(y * self.dot_product(x)))

        self.param[:] = (1 - step_size * mu) * self.param + step_size * mu * wp
        for (k, v) in x.items():
            self.param[k] -= step_size * (-1 * p * y * v)

    def strsaga_step(self, training_point, step_size, mu):
        """ updates the model using strsaga method

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        """

        (i, x, y) = training_point
        p = 1. / (1 + numpy.exp(y * self.dot_product(x)))
        g = -1 * p * y
        alpha = self.table[i] if i in self.table else 0
        m = len(self.table) if len(self.table) != 0 else 1

        self.param[:] = (1 - step_size * mu) * self.param
        for (k, v) in x.items():
            self.param[k] -= step_size * (g - alpha) * v
        self.param -= step_size * (1. / m) * self.table_sum

        self.table[i] = g

        for (k, v) in x.items():
            self.table_sum[k] += (g - alpha) * v

    def strsaga_step_biased(self, training_point, step_size, mu, wp):
        """ updates the model using strsaga method

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        """

        (i, x, y) = training_point
        p = 1. / (1 + numpy.exp(y * self.dot_product(x)))
        g = -1 * p * y
        alpha = self.table[i] if i in self.table else 0
        m = len(self.table) if len(self.table) != 0 else 1

        self.param[:] = (1 - step_size * mu) * self.param + step_size * mu * wp
        for (k, v) in x.items():
            self.param[k] -= step_size * (g - alpha) * v
        self.param -= step_size * (1. / m) * self.table_sum

        self.table[i] = g

        for (k, v) in x.items():
            self.table_sum[k] += (g - alpha) * v

    def predict(self, x, predict_threshold=0.5):
        """ predicts the label for the given input x

        :param x: dict : {int: float}
            feature set for an input data point
        :param threshold: float
            cut off threshold for prediction (default 0.5)

        :return:
            predicted label for the input either 1 or -1
        """

        return 1 if expit(self.dot_product(x)) > predict_threshold else -1

    def loss(self, data):
        """ logistic loss for the given data

        :param data: list of tuples : [(int, {int:float}, int)]
            a list of data points

        :return: float
            logistic loss
        """

        if len(data) == 0:
            return 0
        return sum(numpy.log(1 + numpy.exp(-1 * y * self.dot_product(x))) for (i, x, y) in data) / len(data)

    def reg_loss(self, data, mu):
        """ L2-regularized logistic loss for the given data

        :param data: list of tuples : [(int, {int:float}, int)]
            a list of data points [(index, features, label)]
        :param mu: float
            L2-regularization const

        :return: float
            returns L2-regularized logistic loss for the given data
        """
        if len(data) == 0:
            return 0
        return self.loss(data) + 0.5 * mu * numpy.dot(self.param, self.param)

    def zero_one_loss(self, data, predict_threshold=0.5):
        """ misclassification loss for the given data

        :param data: list of tuples [(int, {int:float}, int)]
            a list of data points
        :param threshold: float
            cut off threshold for prediction (default 0.5)

        :return: float
            returns the misclassification loss for the given data
        """

        if len(data) == 0:
            return 0
        return sum(self.predict(x, predict_threshold) != y for (i, x, y) in data) * 1.0 / len(data)

    def update_effective_set(self, new_buffer_pointer):
        """ updates the effective sample set

        :param new_buffer_pointer: int
            new pointer for the ending pointer
        """

        lst = list(self.T_pointers)
        lst[1] = new_buffer_pointer
        self.T_pointers = (lst[0], lst[1])

    def update_perf(self, current_perf):
        """ updates the current, previous and best observed performances of the model

        :param current_perf:
            current performance of the model
        """
        (current, previous, best) = self.perf
        previous = current
        current = current_perf
        if not best or current < best:
            best = current
        self.perf = (current, previous, best)

        self.sliding_window.append(current_perf)
        if len(self.sliding_window) > self.r:
            self.sliding_window.pop(0)

    def get_current_perf(self):
        """ returns the current performance of the model

        :return:
            current performance of the model
        """
        (current, previous, best) = self.perf
        return current

    def get_best_perf(self):
        """ returns the best observed performance of the model

        :return:
            best observed performance
        """
        (current, previous, best) = self.perf
        return best

    def get_avg_sliding_window(self):
        return mean(self.sliding_window)

    def get_weight(self):
        """ returns the weight of the model

        :return:
            weight of the model
        """
        return self.weight

    def update_weight(self, weight):
        """ updates the weight of the model to the given weight

        :param weight:
        """
        self.weight = weight


class LogisticRegression_DriftSurf:
    """
    A class to define DriftSurf Method

    """

    REG = 'reg'
    ACC = 'zero-one'
    DEFAULT_WEIGHT = 0.5
    GREEDY = 'greedy'

    def __init__(self, d, opt, delta, loss_fn, r, reactive_method='greedy', condition1='best_observed_perf', condition_switch='compare_trained'):
        """

        :param d:
        :param opt:
        :param delta:
        :param loss_fn:
        """
        self.d = d
        self.opt = opt
        self.stable = True
        self.loss_fn = loss_fn
        self.delta = delta
        self.sample_reactive = []
        self.reactive_method = reactive_method

        self.conition1 = condition1
        self.condition_switch = condition_switch

        self.r = r
        self.expert_predictive = LogisticRegression_expert(numpy.random.rand(d), self.opt, r=self.r)
        self.expert_stable = None
        self.expert_reactive = None
        self.expert_frozen = None

    def get_stable(self):
        """ returns if the algorithm is in its sable state

        :return: bool
        """
        return self.stable

    def update_perf_all(self, data, mu):
        """ updates performance of all the present models

        :param data: list of tuples : [(int, {int:float}, int)]
        :param mu: float
        """

        current_perf = self.expert_predictive.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_predictive.zero_one_loss(data)
        self.expert_predictive.update_perf(current_perf)

        if self.stable:
            if self.expert_stable:
                current_perf = self.expert_stable.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_stable.zero_one_loss(data)
                self.expert_stable.update_perf(current_perf)
        else:
            current_perf = self.expert_reactive.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_reactive.zero_one_loss(data)
            self.expert_reactive.update_perf(current_perf)

    def predictive_model(self):
        """ returns the predictive model

        :return:
        """
        if not self.stable and self.reactive_method == self.GREEDY and self.expert_reactive.get_current_perf and self.expert_reactive.get_current_perf() < self.expert_predictive.get_current_perf():
            return self.expert_reactive
        else:
            return self.expert_predictive

    def zero_one_loss(self, data, threshold=0.5):
        """ misclassification loss for the given data

        :param data: list of tuples [(int, {int:float}, int)]
            a list of data points
        :param threshold: float
            cut off threshold for prediction (default 0.5)

        :return: float
            returns the misclassification loss for the given data
        """

        return self.predictive_model().zero_one_loss(data, threshold)

    def reg_loss(self, data, mu):
        """ L2-regularized logistic loss for the given data

        :param data: list of tuples : [(int, {int:float}, int)]
            a list of data points
        :param mu: float
            L2-regularization const

        :return: float
            returns L2-regularized logistic loss for the given data
        """
        return self.predictive_model().reg_loss(data, mu)

    def predict(self, x, threshold=0.5):
        """ predicts the label for the given input x

        :param x: dict : {int: float}
            feature set for an input data point
        :param threshold: float
            cut off threshold for prediction (default 0.5)

        :return:
            predicted label for the input either 1 or -1
        """

        return self.predictive_model().predict(x, threshold)

    def enter_reactive_condition1(self):
        """ Check if R_{X_t}(w) > R_b + delta

        :return: bool
        """

        # compare with the best observed performance so far
        if self.conition1 == 'best_observed_perf':
            condition1 = self.expert_predictive.get_current_perf() > self.expert_predictive.get_best_perf() + self.delta
        # compare with the average of a sliding window
        else:
            condition1 = self.expert_predictive.get_current_perf() > self.expert_predictive.get_avg_sliding_window() + self.delta

        if condition1:
            logging.info('condition 1 holds')

        return condition1

    def enter_reactive_condition2(self):
        """ Determined if R_{X_t}(w) > R_{X_t}(w'') + delta'

        :return: bool
        """

        condition2 = (self.expert_stable) and self.expert_predictive.get_current_perf() > self.expert_stable.get_current_perf() + self.delta/2
        if condition2:
            logging.info('condition 2 holds')

        return condition2

    def enter_reactive(self, buffer_pointer, data, mu):
        """ Determines if we need to enter a reactive state or not. If it decides to enter reactive state this method will do the required processes for that

        :param current_pointer: int
        :param data: list of tuples: [(int, {int:float}, int)]
        :param mu: float
        :return: bool
        """

        if self.stable and (self.enter_reactive_condition1() or self.enter_reactive_condition2()):
            self.stable = False
            self.expert_reactive = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, buffer_pointer, LogisticRegression_DriftSurf.DEFAULT_WEIGHT, self.r)
            self.expert_predictive.update_weight(0.5)
            current_perf = self.expert_reactive.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_reactive.zero_one_loss(data)
            self.expert_reactive.update_perf(current_perf)

            self.expert_frozen = LogisticRegression_expert(numpy.copy(self.expert_predictive.param), self.opt)

            return True

        return False

    def switch_after_reactive(self, mu):
        """ Determines if we need to switch to reactive model at the end of a reactvie state or not

        :param mu:
        :return: bool
        """

        # compare with the performance of the frozen model (snapshot of the predictive model before entering the reactive state)
        if self.condition_switch == 'compare_frozen':
            perf_to_compare = self.expert_frozen.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_frozen.zero_one_loss(self.sample_reactive)

        # compare with the performance of the frozen model + delta (snapshot of the predictive model before entering the reactive state)
        elif self.condition_switch == 'compare_frozen-delta_gap':
            perf_to_compare = -1 * self.delta/2 + self.expert_frozen.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_frozen.zero_one_loss(self.sample_reactive)

        # compare with the performance of the predictive model + delta (predictive model is kept training during the reactive state)
        elif self.condition_switch == 'compare_trained-delta-gap':
            perf_to_compare = -1 * self.delta/2 + self.expert_predictive.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_predictive.zero_one_loss(self.sample_reactive)

        # default case: compare with the performance of the predictive model (predictive model is kept training during the reactive state)
        else:
            perf_to_compare = self.expert_predictive.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_predictive.zero_one_loss(self.sample_reactive)

        perf_reactive = self.expert_reactive.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_reactive.zero_one_loss(self.sample_reactive)

        logging.info(('Performance to compare with : {0}, reactive : {1}').format(perf_to_compare, perf_reactive))

        return (perf_to_compare > perf_reactive)

    def exit_reactive(self, buffer_pointer, mu):
        """ The process of exiting a reactive state

        :param buffer_pointer:
        :param mu:
        """

        self.stable = True

        if self.switch_after_reactive(mu):
            self.expert_predictive.__dict__ = self.expert_reactive.__dict__.copy()

            self.expert_stable = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, buffer_pointer, LogisticRegression_DriftSurf.DEFAULT_WEIGHT, self.r)
            self.sample_reactive = []

            logging.info('exit reactive state with new')
        else:
            logging.info('exit reactive state with old')

    def update_reactive_sample_set(self, data):
        """ updates reactive sample set by adding the given data to it

        :param data:
        """
        self.sample_reactive.extend(data)

class LogisticRegression_AUE:
    """
        A class to define AUE Method presented in :
        'Brzezinski, D. and Stefanowski, J.  Reacting to differenttypes of concept drift: The accuracy updated ensemblealgorithm.IEEE Trans. Neural Netw. Learn. Syst, 25(1):81–94, 2013.'

    """
    K = 10
    # K = 2
    EPS = 1e-20
    
    def __init__(self, d, opt):
        self.init_w = numpy.random.rand(d)
        self.d = d
        self.opt = opt
        self.experts = {}
    
    def predict(self, x, predict_threshold=0.5):
        wp = 0
        wn = 0

        for index in self.experts.keys():
            expert, weight = self.experts[index], self.experts[index].get_weight()

            if (expert.predict(x, predict_threshold) == 1):
                wp += weight
            else:
                wn += weight

        return 1 if wp > wn else -1

    def zero_one_loss(self, data, predict_threshold=0.5):
        if len(data) == 0:
            return 0
        return sum(self.predict(x, predict_threshold) != y for (i, x, y) in data) * 1.0 / len(data)
        
    def update_weights(self, test_set):
        p = ( sum(y for (i, x, y) in test_set) + len(test_set) )/( 2*len(test_set) )
        mser = p*(1-p)
        
        for index, expert in self.experts.items():
            mse = 0
            for (i, x, y) in test_set:
                pr = expit(expert.dot_product(x))
                if y == 1:
                    mse += (1-pr)**2
                else:
                    mse += pr**2
            mse = mse/len(test_set)
            self.experts[index].update_weight(1./(mser + mse + LogisticRegression_AUE.EPS))

        if len(self.experts) == LogisticRegression_AUE.K:
            index = min(self.experts.keys(), key=lambda x: self.experts[x].get_weight())
            del self.experts[index]

        T, _, _ = test_set[0]
        self.experts[T] = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, T)
        self.experts[T].update_weight(1./(mser + LogisticRegression_AUE.EPS))
        self.normalize_weights()

    def normalize_weights(self):
        s = sum(self.experts[k].get_weight() for k in self.experts.keys())
        for k in self.experts.keys():
            self.experts[k].update_weight(self.experts[k].get_weight()/s)

class LogisticRegression_Candor:
    """
        A class to define Candor Method presented in :
        'P. Zhao, L.-W. Cai, and Z.-H. Zhou. Handling concept drift via model reuse. Machine Learning,
        430 109:533–568, 2020.'

    """
    K = 25
    # MU = 400
    MU = 1e-3
    ETA = 0.75

    def __init__(self, d, opt):
        self.init_w = numpy.random.rand(d)
        self.d = d
        self.opt = opt
        self.experts = collections.deque(maxlen=LogisticRegression_Candor.K)

    # def predict(self, x, predict_threshold=0.5):
    #     wp = 0
    #     wn = 0
    #     for e in self.experts:
    #         if (e.predict(x, predict_threshold) == 1):
    #             wp += e.get_weight()
    #         else:
    #             wn += e.get_weight()
    #
    #         # e.update_weight(e.get_weight() * numpy.exp(-1 * e.loss([(0, x, y)]) * LogisticRegression_Candor.ETA))
    #     return 1 if wp > wn else -1

    def predict(self, training_point, predict_threshold=0.5):
        wp = 0
        wn = 0
        (i, x, y) = training_point
        for e in self.experts:
            if (e.predict(x, predict_threshold) == 1):
                wp += e.get_weight()
            else:
                wn += e.get_weight()

            e.update_weight(e.get_weight() * numpy.exp(-1 * e.loss([(0, x, y)]) * LogisticRegression_Candor.ETA))
        return 1 if wp > wn else -1

    def update_weights(self, training_point):
        (i, x, y) = training_point

        for e in self.experts:
            e.update_weight(e.get_weight() * numpy.exp(-1 * e.loss([(0, x, y)]) * LogisticRegression_Candor.ETA))
    #
    # def update_weights_batch(self, batch):
    #     for training_point in batch:
    #         (i, x, y) = training_point
    #
    #         for e in self.experts:
    #             e.update_weight(e.get_weight() * numpy.exp(-1 * e.loss([(0, x, y)]) * LogisticRegression_Candor.ETA))

    # def predict(self, x, predict_threshold=0.5):
    #     if len(self.experts) == 0:
    #         return 1
    #
    #     e = self.experts[-1]
    #     return e.predict(x, predict_threshold)
    #
    # def get_weighted_combination(self):
    #     self.normalize_weights()
    #     w = numpy.zeros(self.d)
    #     if len(self.experts) != 0:
    #         w += self.experts[-1].param
    #     return w

    def zero_one_loss(self, data, predict_threshold=0.5):
        if len(data) == 0:
            return 0
        return sum(self.predict((i,x,y), predict_threshold) != y for (i, x, y) in data) * 1.0 / len(data)
        # return sum(self.predict(x, predict_threshold) != y for (i, x, y) in data) * 1.0 / len(data)

    def get_weighted_combination(self):
        self.normalize_weights()
        w = numpy.zeros(self.d)
        for e in self.experts:
            w += e.param * e.get_weight()
        return w

    def normalize_weights(self):
        s = sum(e.get_weight() for e in self.experts)
        for e in self.experts:
            e.update_weight(e.get_weight() / s)

    def reset_weights(self):
        for e in self.experts:
            e.update_weight(1. / len(self.experts))
