"""SGD and SAGA models for Binary Logistic Regression Task in a single or ensemble setting

This script is written for binary logistic regression task and uses two optimization problem solvers: sgd and saga.
Each can be used in single expert or ensemble setting where the results are aggregated using weighted majority votes.
The loss function for the binary classification task is L2-regularized logistic loss

Each training data point needs to follow the following format: (i, x, y)
where i indicates the index of data point, x is a dictionary which contains the feature (key,value) pairs, and finally
y is the label which could be either +1 or -1.

The ensemble class contains different method to add, and drop experts which makes it to be able to recover from any form
of concept drift. Also, different methods are defined to manage the limited computational resources.

This script requires numpy to be installed within the python environment you are running the script in.
"""

import numpy
from scipy.special import expit
from collections import defaultdict
import operator
import logging


class Opt:
    """
    A class used to determine the optimization problem solver: SGD or SAGA

    ...

    Attributes
    ----------
    SGD : str
        Stochastic Gradient Descent
    SAGA : str
        A variance reduced version of SGD
        'Link <http://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives.pdf>'_
    """

    SGD = 'sgd'
    SAGA = 'saga'

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

        :return: int
            number of gradients computed

        """

        if self.opt == Opt.SGD:
            self.sgd_step(training_point, step_size, mu)
            return 1
        elif self.opt == Opt.SAGA:
            self.saga_step(training_point, step_size, mu)
            return 1

    def sgd_step(self, training_point, step_size, mu):
        """updates the model using sgd method in the single model setting

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

    def saga_step(self, training_point, step_size, mu):
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
    opt : str
        optimization problem solver, either SGD or SAGA
    T_pointers : tuples : (int, int)
        pointers to the start and end of the effective sample set of the expert (default is (0,0))
    T_set : list
        a list of data points' indexes used to trained the model
    T_size: int
        size of the effective sample set used to trained the model (default is 0)
    table: dict : {int: float}
        a dictionary mapping the last computed gradient with respect to a data point
    table_sum: ndarray
        an average over all the most recent computed gradients with respect to all data points seen so far
    weight: float
        weight of the expert (default is 1)
    perf: float
        performance of the expert (default is 0.5)
    perf_prev: float
        performance of the expert on the last time step (default is 0.5)


    Methods
    -------
    dot_product(x)
        returns the dot product of input feature set and the model parameter

    sgd_step(training_point, step_size, mu)
        updates the model using sgd method

    saga_step(training_point, step_size, mu)
        updates the model using saga method

    predict(x, threshold=0.5)
        predicts the label for the given input x

    loss(data)
        returns the logistic loss for the given data

    reg_loss(data, mu)
        returns the L2-regularized logistic loss for the given data

    zero_one_loss(data, threshold=0.5)
        returns the misclassification error for the given data

    mean_absolute_percentage_error(data, threshold=0.5)
        returns the mean absolute percentage error for the given data

    expert_effective_set()
        returns the information related to the expert's effective sample set

    update_effective_set(new_pointer, update_start=False, new_set=[], new_T_size=0)
        updates the effective sample set
    """

    def __init__(self, init_param, opt, buffer_pointer=0):
        """

        :param init_w: ndarray
            parameter to initialize the model's parameter
        :param opt: str
            optimization problem solver, either SGD or SAGA
        :param current_pointer:
            the current pointer in the buffer which indicates the start of the effective sample set (default is 0)
        :param weight: float
            weight of the expert (default is 1)
        :param perf: float
            performance of the expert (default is 0.5)
        """

        self.param = init_param
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

    def saga_step(self, training_point, step_size, mu):
        """ updates the model using saga method

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
            a list of data points
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

        :param new_pointer: int
            new pointer for the ending pointer
        """

        lst = list(self.T_pointers)
        lst[1] = new_buffer_pointer
        self.T_pointers = (lst[0], lst[1])

    def update_perf(self, current_perf):
        (current, previous, best) = self.perf
        previous = current
        current = current_perf
        if not best or current < best:
            best = current
        self.perf = (current, previous, best)

    def get_current_perf(self):
        (current, previous, best) = self.perf
        return current

    def get_best_perf(self):
        (current, previous, best) = self.perf
        return best

class LogisticRegression_DSURF:
    """
    A class to define DriftSurf Method

    Attributes
    ----------


    Methods
    -------

    """

    REG = 'reg'
    ACC = 'zero-one'

    def __init__(self, d, opt, delta, loss_fn):
        """
        :param init_param:
        :param opt:
        :param current_time:
        :param current_pointer
        """

        # self.init_w = numpy.random.rand(d)
        # self.opt = opt


        # self.init_param = numpy.random.rand(d)
        self.d = d
        self.opt = opt
        self.stable = True
        self.loss_fn = loss_fn
        self.delta = delta
        self.sample_reactive = []

        self.expert_predictive = LogisticRegression_expert(numpy.random.rand(d), self.opt)
        self.expert_stable = None
        self.expert_reactive = None
        self.expert_frozen = None #LogisticRegression_expert(self.init_param, self.opt),


    def get_stable(self):
        """

        :return:
        """
        return self.stable

    def update_perf_all(self, data, mu):
        """

        :param data:
        :param predict_threshold:
        :return:
        """

        current_perf = self.expert_predictive.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DSURF.REG else self.expert_predictive.zero_one_loss(data)
        self.expert_predictive.update_perf(current_perf)

        if self.stable:
            if self.expert_stable:
                current_perf = self.expert_stable.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DSURF.REG else self.expert_stable.zero_one_loss(data)
                self.expert_stable.update_perf(current_perf)
        else:
            current_perf = self.expert_reactive.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DSURF.REG else self.expert_reactive.zero_one_loss(data)
            self.expert_reactive.update_perf(current_perf)


    def predictive_model(self):
        """

        :return:
        """

        if not self.stable and self.expert_reactive.get_current_perf and self.expert_reactive.get_current_perf() < self.expert_predictive.get_current_perf():
            return self.expert_reactive
        else:
            return self.expert_predictive


    def zero_one_loss(self, data, threshold=0.5):
        """

        :param data:
        :param threshold:
        :return:
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
        """

        :param x:
        :param threshold:
        :return:
        """
        return self.predictive_model().predict(x, threshold)


    def enter_reactive_condition1(self):

        condition1 = self.expert_predictive.get_current_perf() > self.expert_predictive.get_best_perf() + self.delta
        if condition1:
            logging.info('condition 1 holds')

        return condition1


    def enter_reactive_condition2(self):

        condition2 = (self.expert_stable) and self.expert_predictive.get_current_perf() > self.expert_stable.get_current_perf() + self.delta/2
        if condition2:
            logging.info('condition 2 holds')

        return condition2


    def enter_reactive(self, buffer_pointer, data, mu):
        """

        :param current_pointer:
        :param data:
        :param mu:
        :return:
        """

        if self.stable and (self.enter_reactive_condition1() or self.enter_reactive_condition2()):
            self.stable = False
            self.expert_reactive = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, buffer_pointer)
            current_perf = self.expert_reactive.reg_loss(data, mu) if self.loss_fn == LogisticRegression_DSURF.REG else self.expert_reactive.zero_one_loss(data)
            self.expert_reactive.update_perf(current_perf)

            self.expert_frozen = LogisticRegression_expert(numpy.copy(self.expert_predictive.param), self.opt)

            return True

        return False


    def switch_after_reactive(self, mu):

        perf_frozen = self.expert_frozen.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DSURF.REG else self.expert_frozen.zero_one_loss(self.sample_reactive)
        perf_reactive = self.expert_reactive.reg_loss(self.sample_reactive, mu) if self.loss_fn == LogisticRegression_DSURF.REG else self.expert_reactive.zero_one_loss(self.sample_reactive)

        logging.info(('Performance of frozen : {0}, reactive : {1}').format(perf_frozen, perf_reactive))

        return (perf_frozen > perf_reactive)


    def exit_reactive(self, buffer_pointer, mu):
        """

        :param predict_threshold:
        :return:
        """

        self.stable = True

        if self.switch_after_reactive(mu):
            self.expert_predictive.__dict__ = self.expert_reactive.__dict__.copy()

            self.expert_stable = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, buffer_pointer)
            self.sample_reactive = []

            logging.info('exit reactive state with new')
        else:
            logging.info('exit reactive state with old')


    def update_reactive_sample_set(self, data):
        """

        :param data:
        :return:
        """
        self.sample_reactive.extend(data)


class LogisticRegression_AUE:
    K = 10
    EPS = 1e-20
    
    def __init__(self, d, opt):
        self.init_w = numpy.random.rand(d)
        self.d = d
        self.opt = opt
        self.experts = {}
        self.weights = {}
        # self.T1 = {}
    
    def predict(self, x, predict_threshold=0.5):
        wp = 0
        wn = 0

        for index in self.experts.keys():
            expert, weight = self.experts[index], self.weights[index]

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
            self.weights[index] = 1./(mser + mse + LogisticRegression_AUE.EPS)
            
        if len(self.experts) == LogisticRegression_AUE.K:
            index = min(self.weights, key=self.weights.get)
            del self.experts[index]
            del self.weights[index]
        
        T, _, _ = test_set[0]
        self.experts[T] = LogisticRegression_expert(numpy.random.rand(self.d), self.opt, T)
        self.weights[T] = 1./(mser + LogisticRegression_AUE.EPS)
        # self.T1[T] = T
        self.normalize_weights()

    def normalize_weights(self):
        s = sum(w for w in self.weights.values())
        for k in self.weights.keys():
            self.weights[k] /= s