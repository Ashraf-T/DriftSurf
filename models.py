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

# default number of time steps
b = 100

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
        based on the optimization problem solver calls the proper updating method

    sgd_step(training_point, step_size, mu)
        updates the model using sgd method in single model setting

    saga_step(training_point, step_size, mu)
        updates the model using saga method in single model setting

    sgd_step(training_point, step_size, mu, key)
        updates the model using sgd method in ensemble setting

    saga_step(training_point, step_size, mu, key)
        updates the model using saga method in ensemble setting

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
        """updates the model using sgd method in single model setting

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
        """updates the model using saga method in single model setting

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

    def sgd_step(self, training_point, step_size, mu, key):
        """updates the model using sgd method in ensemble setting

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        :param key: int
                    expert index in the ensemble

        :raise NotImplementedError
            This abstract method will be override in the derived classes
        """

        raise NotImplementedError()

    def saga_step(self, training_point, step_size, mu, key):
        """updates the model using saga method in ensemble setting

        :param training_point: tuple : (int, {int:float}, int)
            training data point in form of (i, x, y) where i is its index, x is a dictionary of the
            features (key, value) pairs, and y is the label which can be either 1 or -1
        :param step_size: float
            step size used to update the model
        :param mu: float
            L2-regularization const
        :param key: int
                    expert index in the ensemble

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

    def __init__(self, init_w, opt, current_pointer=0, weight=1, perf=0.5):
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

        self.param = numpy.copy(init_w)
        self.opt = opt

        self.T_pointers = (current_pointer, current_pointer)
        self.T_set = []
        self.T_size = 0

        self.table = {}
        self.table_sum = numpy.zeros(self.param.shape)

        self.weight = weight
        self.perf = perf
        self.perf_prev = perf


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

    def mean_absolute_percentage_error(self, data, threshold=0.5):
        """ mean absolute percentage error for the given data

        :param data: list of tuples : [(int, {int:float}, int)]
            a list of data points
        :param threshold: float
            cut off threshold for prediction (default 0.5)

        :return: float
            returns the mean absolute percentage error for the given data
        """
        out = 0
        for (i, x, y) in data:
            out += (y - self.predict(x, threshold=threshold))/y
        return out*100./len(data)

    def update_weight(self, new_weight):
        """

        :param new_weight:
        """

        self.weight = new_weight

    def expert_effective_set(self):
        """ returns the information related to the expert's effective sample set

        :return: tuples : ((int, int), list, int)
            returns the pointers to the begining and end indeces of the effective sample set along with the effective
            sample set and its size
        """

        return self.T_pointers, self.T_set, self.T_size

    def update_effective_set(self, new_pointer, update_start=False, new_set=[], new_T_size=0):
        """ updates the effective sample set

        :param new_pointer: int
            new pointer for the ending pointer
        :param update_start: bool
            determines if you need to update the starting pointer too or not (default = False)
        :param new_set: list
            updated effective sample set (default = [])
        :param new_T_size: int
            size of the updated effective sample set (default = 0)
        """

        lst = list(self.T_pointers)
        lst[1] = new_pointer
        if update_start:
            lst[0] = new_pointer
        self.T_pointers = (lst[0], lst[1])
        self.T_size = new_T_size
        self.T_set = numpy.copy(new_set)


class LogisticRegression_StableReactive(Model):
    """
    A class to define stable-reactive states which is a children of Model

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, init_param, opt, current_pointer, delta, loss = 'reg'):
        """
        :param init_param:
        :param opt:
        :param current_time:
        :param current_pointer
        """

        self.init_param = init_param
        self.opt = opt
        self.pointer_buff = current_pointer
        self.stable = True # stable or reactive
        self.delta = delta
        self.sample_reactive = []

        self.loss_fn = loss

        self.experts = {
                        'main' :
                            {
                                'expert' : LogisticRegression_expert(numpy.copy(init_param), self.opt, current_pointer),
                                'best_observed_perf' : None,
                                'prev_perf' : None,
                                'current_perf' : None
                            },
                        'new_stable' :
                            {
                                'expert' : None,
                                'best_observed_perf' : None,
                                'prev_perf': None,
                                'current_perf': None
                            },
                        'new_reactive' :
                            {
                                'expert' : None,
                                'best_observed_perf' : None,
                                'prev_perf' : None,
                                'current_perf': None
                            },
                        'frozen':
                            {
                                'expert' : LogisticRegression_expert(numpy.copy(init_param), self.opt, current_pointer),
                                'best_observed_perf': None,
                                'prev_perf': None,
                                'current_perf': None
                            }
                         }
        # self.reactive_frozen_expert = None


    def get_stable(self):
        """

        :return:
        """
        return self.stable


    def create_new_expert(self, current_pointer):
        """

        :param current_pointer:
        :return:
        """

        if self.stable:
            expert = 'new_stable'
        else:
            expert = 'new_reactive'

        self.experts[expert]['expert'] = LogisticRegression_expert(numpy.copy(self.init_param), self.opt, current_pointer)
        self.experts[expert]['current_perf'] = None
        self.experts[expert]['prev_perf'] = None
        self.experts[expert]['best_observed_perf'] = None



    def update_best_observed_perf(self):
        """

        :return:
        """
        for key in self.experts.keys():
            if self.experts[key]['expert'] != None:
                if self.experts[key]['best_observed_perf'] == None:
                    self.experts[key]['best_observed_perf'] = self.experts[key]['current_perf']
                else:
                    self.experts[key]['best_observed_perf'] = min(self.experts[key]['best_observed_perf'], self.experts[key]['current_perf'])


    def predictive_model(self):
        """

        :return:
        """

        if self.experts['new_reactive']['expert'] != None and self.experts['new_reactive']['current_perf'] != None and self.experts['main']['current_perf'] > self.experts['new_reactive']['current_perf']:
            return 'new_reactive'
        else:
            return 'main'

    def update_perf_all(self, data, mu):
        """

        :param data:
        :param predict_threshold:
        :return:
        """

        for key in self.experts.keys():
            if self.experts[key]['current_perf'] != None:
                self.experts[key]['prev_perf'] = self.experts[key]['current_perf']

            if self.experts[key]['expert'] != None:
                if self.loss_fn == 'reg':
                    self.experts[key]['current_perf'] = self.experts[key]['expert'].reg_loss(data, mu)
                else:
                    self.experts[key]['current_perf'] = self.experts[key]['expert'].zero_one_loss(data)


    def update_perf(self, key, data, mu):
        """

        :param data:
        :param predict_threshold:
        :return:
        """

        if self.experts[key]['current_perf'] != None:
            self.experts[key]['prev_perf'] = self.experts[key]['current_perf']

        if self.loss_fn == 'reg':
            self.experts[key]['current_perf'] = self.experts[key]['expert'].reg_loss(data, mu)
        else:
            self.experts[key]['current_perf'] = self.experts[key]['expert'].zero_one_loss(data)



    def zero_one_loss(self, data, threshold=0.5):
        """

        :param data:
        :param threshold:
        :return:
        """

        return self.experts[self.predictive_model()]['expert'].zero_one_loss(data, threshold)


    def reg_loss(self, data, mu):
        """ L2-regularized logistic loss for the given data

        :param data: list of tuples : [(int, {int:float}, int)]
            a list of data points
        :param mu: float
            L2-regularization const

        :return: float
            returns L2-regularized logistic loss for the given data
        """
        return self.experts[self.predictive_model()]['expert'].reg_loss(data, mu)


    def predict(self, x, threshold=0.5):
        """

        :param x:
        :param threshold:
        :return:
        """
        return self.experts[self.predictive_model()]['expert'].predict(x, threshold)


    def update_buffer_pointer(self, pointer_buff):
        """

        :param pointer_buff:
        :return:
        """
        self.pointer_buff = pointer_buff


    def enter_reactive_condition1(self):

        condition1 = (self.experts['main']['current_perf'] > self.experts['main']['best_observed_perf'] + self.delta)
        if condition1:
            logging.info('condition 1')

        return condition1


    def enter_reactive_condition2(self):

        condition2 = (self.experts['new_stable']['expert']!=None and self.experts['main']['current_perf'] > self.experts['new_stable']['current_perf'] + self.delta/2)

        if condition2:
            logging.info('condition 2')

        return condition2


    def enter_reactive(self, current_pointer, data, mu):
        """

        :param current_pointer:
        :param data:
        :param mu:
        :return:
        """
        if self.stable and ( self.enter_reactive_condition1() or self.enter_reactive_condition2()):

            self.stable = False

            self.create_new_expert(current_pointer)
            self.update_perf('new_reactive', data, mu) # we need to know the current perf of this expert to make the greedy decision next time step

            self.experts['frozen']['expert'] = LogisticRegression_expert(numpy.copy(self.experts['main']['expert'].param), self.opt, current_pointer)
            # self.experts['frozen']['expert'].__dict__ = self.experts['main']['expert'].__dict__.copy()

            return True

        return False


    def switch_after_reactive(self, mu):

        if self.loss_fn == 'reg':
            perf_frozen = self.experts['frozen']['expert'].reg_loss(self.sample_reactive, mu)
            perf_new = self.experts['new_reactive']['expert'].reg_loss(self.sample_reactive, mu)
        else:
            perf_frozen = self.experts['frozen']['expert'].zero_one_loss(self.sample_reactive)
            perf_new = self.experts['new_reactive']['expert'].zero_one_loss(self.sample_reactive)

        logging.info(('frozen {0}, new {1}').format(perf_frozen, perf_new))

        return (perf_frozen > perf_new)


    def exit_reactive(self, current_pointer, mu):
        """

        :param predict_threshold:
        :return:
        """


        ret_expert = 'old' # default decision is to choose the old one, unless the new one performs better

        if self.switch_after_reactive(mu):
            self.experts['main']['expert'].__dict__ = self.experts['new_reactive']['expert'].__dict__.copy()
            self.experts['main']['current_perf'] = self.experts['new_reactive']['current_perf']
            self.experts['main']['prev_perf'] = self.experts['new_reactive']['prev_perf']
            self.experts['main']['best_observed_perf'] = self.experts['new_reactive']['best_observed_perf']
            ret_expert = 'new'

        self.stable = True
        self.create_new_expert(current_pointer)
        # self.update_perf('new_stable', data, predict_threshold)
        self.sample_reactive = []
        logging.info('exit reactive state with {0}'.format(ret_expert))


    def update_reactive_sample_set(self, data):
        """

        :param data:
        :return:
        """
        self.sample_reactive.extend(data)



class LogisticRegression_Ensemble(Model):
    """
    A class to define an ensmble of experts which is a children of Model
    ...

    Attributes
    ----------



    Methods
    -------

    """

    def __init__(self, init_param, opt, current_time=0,  max_time=100, current_pointer=0, carry_over_function=False, generate_multiple_experts=False, weight_of_experts=[1], perf_of_experts=[0.5], number_of_experts=1, starting_id=1):
        """

        :param init_param:
        :param opt:
        :param current_time:
        :param max_time:
        :param current_pointer:
        :param carry_over_function:
        :param generate_multiple_experts:
        :param weight_of_experts:
        :param perf_of_experts:
        :param number_of_experts:
        :param starting_id:
        """

        self.opt = opt
        self.experts = {}
        self.timeAD = {}
        self.weight_over_time = defaultdict(list)
        self.perf_over_time = defaultdict(list)

        self.max_time = max_time
        self.best_id = 1
        self.dim = len(init_param)

        self.max_id = 1
        self.pointer_buff = current_pointer

        self.perf = 0.5
        self.perf_prev = 0.5


        self.experts[self.max_id] = LogisticRegression_expert(numpy.random.rand(self.dim), self.opt, self.pointer_buff,1, 0.5)
        self.timeAD[self.max_id] = (current_time, self.max_time)
        # self.weight_over_time[self.max_id] = [weight]
        # self.perf_over_time[self.max_id] = [perf]
        print("time : " + str(current_time) + " adding a new expert with random initialization - " + str(self.opt) + " : " + str(self.max_id))
        self.max_id += 1
        # self.create_new_expert(current_time)
        # for index in range(number_of_experts):
        # self.experts[index+starting_id] = LogisticRegression_expert(init_param, opt, current_pointer, weight_of_experts[index], perf_of_experts[index])
        # self.timeAD[index+starting_id] = (current_time, self.max_time)
        # self.weight_over_time[index+starting_id] = [self.experts[index+starting_id].weight]
        # self.perf_over_time[index+starting_id] = [self.experts[index+starting_id].perf]
        # self.max_id += 1

    def create_new_expert(self, current_time, current_pointer):
        """

        :param current_time:
        :return:
        """

        if len(self.experts.keys()) > 0:

            weight = self.experts[self.best_id].weight
            perf = self.experts[self.best_id].perf

            self.experts[self.max_id] = LogisticRegression_expert(numpy.copy(self.experts[self.best_id].param), self.opt, current_pointer, weight, perf)
            self.timeAD[self.max_id] = (current_time, self.max_time)
            self.weight_over_time[self.max_id] = [weight]
            self.perf_over_time[self.max_id] = [perf]
            print("time : " + str(current_time) + " adding a new expert carrying over the previously best function - " + str(self.opt) + " : " + str(self.max_id))
            self.max_id += 1
        else:
            weight, perf = 1, 0.5

        self.experts[self.max_id] = LogisticRegression_expert(numpy.random.rand(self.dim), self.opt, current_pointer,weight, perf)
        self.timeAD[self.max_id] = (current_time, self.max_time)
        self.weight_over_time[self.max_id] = [weight]
        self.perf_over_time[self.max_id] = [perf]
        print("time : " + str(current_time) + " adding a new expert with random initialization - " + str(self.opt) + " : " + str(self.max_id))
        self.max_id += 1

        self.normalize_weight()


    def normalize_weight(self):
        """

        :param norm:
        :return:
        """
        # if norm == 'sum':
        value = self.sum_of_weights()
        if value != 0:
            for key in self.experts.keys():
                self.experts[key].weight /= value

    def sum_of_weights(self):
        value = 0.0
        for key in self.experts.keys():
            value += self.experts[key].weight
        return value


    def update_perf(self, data, predict_threshold=0.5):
        for key in self.experts.keys():
            self.experts[key].perf_prev = self.experts[key].perf
            self.experts[key].perf = 1 - self.experts[key].zero_one_loss(data, predict_threshold)
            self.experts[key].weight = self.experts[key].perf

        # normalize weights
        self.normalize_weight()
        for key in self.experts.keys():
            self.weight_over_time[key].append(self.experts[key].weight)
            self.perf_over_time[key].append(self.experts[key].perf)
            # print key, self.experts[key].perf


    def find_best_expert(self):
        self.perf_prev = self.experts[self.best_id].perf_prev
        max_perf = 0
        index = 1
        for key in self.experts.keys():
            if max_perf < self.experts[key].perf:
                max_perf = self.experts[key].perf
                index = key

        self.best_id = index
        self.perf = self.experts[index].perf
        return index

    def add_expert(self, current_time, eta=0.1, current_pointer = 0):

        # if len(self.experts.keys()) == 0:
        #     print 'length zero -> add new expert'
        #     self.create_new_expert(current_time)

        if self.perf < self.perf_prev - eta:
            self.create_new_expert(current_time, current_pointer)

        # self.normalize_weight()


    def zero_one_loss(self, data, threshold=0.5):
        if len(data) == 0:
            return 0
        return sum(self.predict(x, threshold) != y for (i, x, y) in data) * 1.0 / len(data)

    def predict(self, x, threshold=0.5):
        return self.experts[self.best_id].predict(x, threshold)

    # def update_weights(self):
    #     """
    #
    #     :param data:
    #     :param current_time:
    #     :param predict_threshold:
    #     :param weight_update_strategy:
    #     :param drop_threshold:
    #     :param warmup_period:
    #     :param drop:
    #     :param drop_method:
    #     :param add_method:
    #     :param eta:
    #     :param norm:
    #     :return:
    #     """
    #
    #     for key in self.experts.keys():
    #         self.experts[key].weight = self.experts[key].perf
    #
    #     # normalize weights
    #     self.normalize_weight()
    #     for key in self.experts.keys():
    #         self.weight_over_time[key].append(self.experts[key].weight)
    #         self.perf_over_time[key].append(self.experts[key].perf)
    #         # print key, self.experts[key].perf

    def update_buffer_pointer(self, pointer_buff):
        """

        :param pointer_buff:
        :return:
        """
        self.pointer_buff = pointer_buff

    def remove_expert(self, current_time, drop_threshold=0.5, warmup_period=5): #threshold is about miss classification rate
        """

        :param current_time:
        :param drop_threshold:
        :param warmup_period:
        :param drop:
        :param drop_method:
        :return:
        """
        for key in self.experts.keys():
            start_time, end_time = self.timeAD[key]
            if self.experts[key].weight < drop_threshold and current_time - start_time > warmup_period and key != self.best_id:
                print("time : " + str(current_time) + " removing an expert - " + str(self.opt) + " : " + str(key))

                del self.experts[key]
                lst = list(self.timeAD[key])
                lst[1] = current_time
                self.timeAD[key] = (lst[0], lst[1])
                # del self.weight_over_time[key][len(self.weight_over_time[key])-1]
                # del self.perf_over_time[key][len(self.perf_over_time[key])-1]

        # elif drop_method != 'noDrop':
        #     print "undefined drop method"

        self.normalize_weight()













