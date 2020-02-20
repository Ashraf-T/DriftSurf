import numpy
from scipy.special import expit
import logging


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

    def __init__(self, init_param, opt, buffer_pointer=0, weight=1):
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

        self.perf = (None, None, None) # current, previous, best observed

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

    def __init__(self, d, opt, delta, loss_fn, method='greedy'):
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
        self.method = method

        self.expert_predictive = LogisticRegression_expert(numpy.random.rand(d), self.opt)
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
        if not self.stable and self.method == self.GREEDY and self.expert_reactive.get_current_perf and self.expert_reactive.get_current_perf() < self.expert_predictive.get_current_perf():
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

        condition1 = self.expert_predictive.get_current_perf() > self.expert_predictive.get_best_perf() + self.delta
        # if condition1:
        #     logging.info('condition 1 holds')

        return condition1

    def enter_reactive_condition2(self):
        """ Determined if R_{X_t}(w) > R_{X_t}(w'') + delta'

        :return: bool
        """

        condition2 = (self.expert_stable) and self.expert_predictive.get_current_perf() > self.expert_stable.get_current_perf() + self.delta/2
        # if condition2:
        #     logging.info('condition 2 holds')

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
            self.expert_reactive = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, buffer_pointer, LogisticRegression_DriftSurf.DEFAULT_WEIGHT)
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

        perf_frozen = self.expert_frozen.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_frozen.zero_one_loss(self.sample_reactive)
        perf_reactive = self.expert_reactive.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DriftSurf.REG else self.expert_reactive.zero_one_loss(self.sample_reactive)

        # logging.info(('Performance of frozen : {0}, reactive : {1}').format(perf_frozen, perf_reactive))

        return (perf_frozen > perf_reactive)

    def exit_reactive(self, buffer_pointer, mu):
        """ The process of exiting a reactive state

        :param buffer_pointer:
        :param mu:
        """

        self.stable = True

        if self.switch_after_reactive(mu):
            self.expert_predictive.__dict__ = self.expert_reactive.__dict__.copy()

            self.expert_stable = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, buffer_pointer, LogisticRegression_DriftSurf.DEFAULT_WEIGHT)
            self.sample_reactive = []

        #     logging.info('exit reactive state with new')
        # else:
        #     logging.info('exit reactive state with old')

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
