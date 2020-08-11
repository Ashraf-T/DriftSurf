import models
import hyperparameters
import random
import numpy
import logging
import copy

from data_structures.attribute_scheme import AttributeScheme
from readers.arff_reader import ARFFReader
from drift_detection.__init__ import *
import classifier
# from classifier.__init__ import *
from dictionary.tornado_dictionary import TornadoDic
from filters.attribute_handlers import *

class Training:

    LIMITED = 'all_models'
    UNLIMITED = 'each_model'

    def __init__(self, dataset, base_learner, computation='each_model', rate=2, algo_names=['Aware', 'OBL', 'MDDM', 'AUE', 'DriftSurf'], drift=True):

        self.algorithms = algo_names
        self.loss = {}
        self.pointers = {}
        self.computation = computation.lower()
        self.rate = rate

        self.dataset_name = dataset
        self.Y, self.attributes, X = ARFFReader.read("data/synthetic/{0}.arff".format(self.dataset_name))
        self.attributes_scheme = AttributeScheme.get_scheme(self.attributes)

        self.learner = getattr(classifier, base_learner)(self.Y, self.attributes_scheme['nominal'])
        self.data = self.data_transform(X)
        self.n = len(self.data)

        if self.dataset_name.startswith('sea'):
            self.dataset_name = 'sea'

        self.drift_times = hyperparameters.DRIFT_TIMES[self.dataset_name]
        self.mu = hyperparameters.MU[self.dataset_name]
        self.step_size = hyperparameters.STEP_SIZE[self.dataset_name]
        self.b = hyperparameters.b[self.dataset_name] if self.dataset_name in hyperparameters.b.keys() else hyperparameters.b['default']
        self.lam = int(self.n//self.b)
        self.rho = self.lam * self.rate

    def data_transform(self, data):

        processed_data = []
        for i in range(len(data)):
            r = copy.copy(data[i])
            for k in range(0, len(r) - 1):
                if self.learner.LEARNER_CATEGORY == TornadoDic.NOM_CLASSIFIER and self.attributes[k].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                    r[k] = Discretizer.find_bin(r[k], self.attributes_scheme['nominal'][k])
                elif self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER and self.attributes[k].TYPE == TornadoDic.NOMINAL_ATTRIBUTE:
                    r[k] = NominalToNumericTransformer.map_attribute_value(r[k], self.attributes_scheme['numeric'][k])
            # NORMALIZING NUMERIC DATA
            if self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER:
                r[0:len(r) - 1] = Normalizer.normalize(r[0:len(r) - 1], self.attributes_scheme['numeric'])
            processed_data.append(r)

        return processed_data

    def update_learner_pointers(self, algo, num, start, end):
        self.pointers[algo][num] = (start, end)

    @staticmethod
    def setup_experiment(dataset, rate, n):
        """
            setup the hyperparameters for the experiment
        :param dataset: str
                name of the dataset
        :param rate: int
                rho/lam
        :param n: int
                # of data points in the given dataset
        :return:
                regularization term (mu), step_size (eta), total number of batches (b), size of each batch (lam), computational power (rho)
        """

        mu = hyperparameters.MU[dataset]
        step_size = hyperparameters.STEP_SIZE[dataset]
        b = hyperparameters.b[dataset] if dataset in hyperparameters.b.keys() else hyperparameters.b['default']
        lam = int(n//b)
        rho = lam * rate

        return mu, step_size, b, lam, rho

    def learner_zero_one_loss(self, algo, data):

        if len(data) == 0:
            return 0

        return sum(getattr(self, algo).do_testing(x) != x[-1] for x in data) * 1.0 / len(data)

    def update_loss(self, test_set, time):
        """
            computes and updates the loss of algorithms at time t over the given test set
        :param test_set:
        :param time:
        """

        for algo in self.algorithms:
            if time > 0:

                if algo == 'DriftSurf':
                    self.loss[algo][time] = self.DriftSurf.zero_one_loss(test_set)
                else:
                    self.loss[algo][time] = self.learner_zero_one_loss(algo, test_set)

    def setup_algorithms(self, delta, detector=MDDM_G(), r=None):
        """
            set parameters of algorithms in the begining of the training
        :param delta: float
                DriftSurf's parameter for drift detection
        :param loss_fn: str (zero-one / reg)
                loss function DriftSurf check for performance degrading
        :param detector:
                drift detector method for MDDM
        :param Aware_reset: str
                when to reset Aware: before or after computing the loss at drift times
        :param r:  int
                length of the reactive state in DriftSurf
        """

        for algo in self.algorithms:
            if algo == 'DriftSurf': self.setup_DriftSurf(delta, r)
            elif algo == 'MDDM': self.setup_MDDM(detector)
            elif algo == 'Aware': self.setup_Aware()
            else: getattr(self, 'setup_{0}'.format(algo))()

    def setup_DriftSurf(self, delta, r=None):
        """
            setup parameters of DriftSurf
        :param delta: float
                delta-degration in performance is considered as a sign of drift
        :param loss_fn: str (reg, zero-one)
                loss function that DriftSurf check for performance degration
        :param r: int
                length of the reactive state
        """

        self.DriftSurf_t = 0
        self.DriftSurf_r = r if r else (hyperparameters.r[self.dataset_name] if self.dataset_name in hyperparameters.r.keys() else hyperparameters.r['default'])
        self.DriftSurf = models.DriftSurf_general(copy.copy(self.learner), delta)

    def setup_MDDM(self, detector=MDDM_G()):
        """
            setup parameters of MDDM
        :param detector: (MDDM-A, MDDM-E, MDDM-G)
                drift detector of MDDM, defualt is set to be MDDM-D()
        """
        self.MDDM = copy.copy(self.learner)
        self.MDDM_drift_detector = detector

    def setup_AUE(self):
        """
            setup AUE
        """
        self.AUE = copy.copy(self.learner)

    def setup_Aware(self):
        """
            setup Aware
        :param reset: str (B / A)
                when to reset parameters of the predictive model in Aware: before computing loss or after - default is set to be before
        """
        self.Aware = copy.copy(self.learner)

    def setup_OBL(self):
        """
            setup oblivious algorithm
        """
        self.OBL = copy.copy(self.learner)

    def setup_Candor(self):
        """
            setup Candor
        """
        self.Candor = copy.copy(self.learner)

    def process_MDDM(self, time, new_batch):
        """
            MDDM's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """

        if self.MDDM.is_ready() and (self.MDDM_drift_detector.test(self.MDDM, new_batch) and time != 0):
            self.MDDM.reset()
            self.update_learner_pointers('MDDM', 0, time*self.lam, time*self.lam)
            self.MDDM_drift_detector.reset()
            logging.info('MDDM drift detected, reset model : {0}'.format(time))

        self.process_strsaga_style('MDDM', 0)

    def process_AUE(self, time, new_batch):
        """
            AUE's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        pass
        #TODO: fix this one
        # self.AUE.update_weights(new_batch)
        # logging.info('AUE Experts at time {0}: {1}'.format(time, [int(k / self.lam) for k in self.AUE.experts.keys()]))
        # for index, expert in self.AUE.experts.items():
        #     getattr(self, 'update_{0}_model'.format(self.opt))(expert)

    def process_Candor(self, time, new_batch):
        """
            Candor's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        pass
        #TODO: fix this function
        #
        # wp = self.Candor.get_weighted_combination()
        # expert = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S)  # alt: first arg is wp
        # if time == 0:
        #     getattr(self, 'update_{0}_model'.format(self.opt))(expert)
        # else:
        #     getattr(self, 'update_{0}_model_biased'.format(self.opt))(expert, wp)
        #
        # self.Candor.experts.append((expert, wp))
        # self.Candor.reset_weights()

    def process_DriftSurf(self, time, new_batch):
        """
            DriftSurf's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """

        self.DriftSurf.update_perf_all(new_batch)

        if self.DriftSurf.stable:
            if self.DriftSurf.enter_reactive(new_batch):
                self.DriftSurf_t = 0
                self.update_learner_pointers('DriftSurf', 2, time*self.lam, time*self.lam) #reactive pointers
                logging.info('DriftSurf enters reactive state : {0}'.format(time))

            else:
                # update models
                self.process_strsaga_style('DriftSurf', 0) # predictive
                if 1 in self.pointers['DriftSurf']:
                    self.process_strsaga_style('DriftSurf', 1) # stable

        if not self.DriftSurf.stable:
            # update models
            self.DriftSurf.update_reactive_sample_set(new_batch)
            self.process_strsaga_style('DriftSurf', 0)  # predictive
            self.process_strsaga_style('DriftSurf', 2)  # reactive
            self.DriftSurf_t += 1
            if self.DriftSurf_t == self.DriftSurf_r :
                self.DriftSurf.exit_reactive()
                self.update_learner_pointers('DriftSurf', 1, time*self.lam, time*self.lam)
                logging.info('DriftSurf exits reactive state : {0}'.format(time))

    def process_strsaga_style(self, algo, num):

        # weight = model.get_weight() if self.computation == Training.LIMITED else 1
        lst = list(self.pointers[algo][num])
        for s in range(int(self.rho)):
            if s % 2 == 0 and lst[1] < self.S + self.lam:
                j = lst[1]
                lst[1] += 1
            else:
                j = random.randrange(lst[0], lst[1])
            point = copy.copy(self.data[j])

            if algo != 'DriftSurf' and getattr(self, algo).LEARNER_TYPE == TornadoDic.TRAINABLE:
                getattr(self, algo).do_training(point)
                getattr(self, algo).set_ready()
            elif algo == 'DriftSurf' and self.DriftSurf.expert_predictive.LEARNER_TYPE == TornadoDic.TRAINABLE:
                if num == 0:
                    self.DriftSurf.expert_predictive.do_training(point)
                    self.DriftSurf.expert_predictive.set_ready()
                elif num == 1:
                    self.DriftSurf.expert_stable.do_training(point)
                    self.DriftSurf.expert_stable.set_ready()
                elif num == 2:
                    self.DriftSurf.expert_reactive.do_training(point)
                    self.DriftSurf.expert_reactive.set_ready()
            else:
                getattr(self, algo).do_loading(point)

        self.update_learner_pointers(algo, num, lst[0], lst[1])

    def process_Aware(self, time, data):

        if time in self.drift_times:
            self.Aware.reset()
            self.update_learner_pointers('Aware', 0, time*self.lam, time*self.lam)
            logging.info('Aware reset model : {0}'.format(time))

        self.process_strsaga_style('Aware', 0)

    def process_OBL(self):
        """
            oblivious algorithm's process
        """
        pass
        #TODO: update this
        # lst = list(self.OBL.T_pointers)
        # for s in range(self.rho):
        #     if s % 2 == 0 and lst[1] < self.S + self.lam:
        #         lst[1] += 1
        #         j = random.randrange(lst[0], lst[1])
        #     point = (j, self.X[j], self.Y[j])
        #     self.OBL.update_step(point, self.step_size, self.mu)
        # self.OBL.update_effective_set(lst[1])

    def process(self, delta=0.1, drift_detectr=MDDM_G(), r=None):
        """
            Train algorithms over the given dataset arrivin in streaming setting over b batches
        :param delta:
                DriftSurf's parameter for drift detection
        :param loss_fn:
                DriftSurf's parameter for drift detection
        :param drift_detectr:
                MDDM's drift detector
        :param Aware_reset:
                When to reset Aware
        :param r:
                Length of the reactive state in DriftSurf
        """

        self.S = 0
        self.setup_algorithms(delta, drift_detectr, r)

        for algo in self.algorithms:
            self.loss[algo] = [0] * self.b
            self.pointers[algo] = {0 : (0, 0)}

        logging.info('dataset : {0}, n : {1}, b : {2}'.format(self.dataset_name, self.n, self.b))

        for time in range(self.b):

            print(time)

            # measure accuracy over upcoming batch
            test_set = [self.data[i] for i in range(self.S, self.S + self.lam)]
            self.update_loss(test_set, time)

            if time in self.drift_times and 'Aware' in self.algorithms:
                self.setup_Aware()

            for algo in self.algorithms:
                if algo in ['SGD', 'OBL']: getattr(self, 'process_{0}'.format(algo))()
                else: getattr(self, 'process_{0}'.format(algo))(time, test_set)


            self.S += self.lam

        return self.loss

