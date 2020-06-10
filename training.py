import models
import read_data as data
import hyperparameters
from drift_detection.__init__ import *
import random
import numpy
import logging


class Training:

    STRSAGA = models.Opt.STRSAGA
    SGD = models.Opt.SGD
    LIMITED = 'algorithm'
    BEFORE = 'B'
    AFTER = 'A'

    def __init__(self, dataset, computation='model', rate=2, base_learner = 'STRSAGA', algo_names=['Aware', 'SGD', 'OBL', 'MDDM', 'AUE', 'DriftSurf']):
        """

        :param dataset: str
                name of the dataset
        :param computation: str (algorithm / model)
                algorithm: rho computational power is given to each algorithm and needs to be divided between learners
                model: rho computational power is given to each model in each algorithm
        :param rate: int
                rate = (rho/lam)
        :param base_learner: (STRSAGA / SGD)
                the base learner used by each algorithm can be either strsaga or sgd
        :param algo_names: ['Aware', 'SGD', 'OBL', 'MDDM', 'AUE', 'DriftSurf']
                list of algorithms we want to train over the given dataset
                Aware: have oracle knowledge about times of drifts
                SGD: single-pass SGD
                OBL: oblivious to drift
                MDDM: presented in 'Pesaranghader, A., Viktor, H. L., and Paquet, E. Mcdiarmiddrift detection methods for evolving data streams.   InIJCNN, pp. 1–9, 2018.'
                AUE: presented in 'Brzezinski, D. and Stefanowski, J.  Reacting to differenttypes of concept drift: The accuracy updated ensemblealgorithm.IEEE Trans. Neural Netw. Learn. Syst, 25(1):81–94, 2013.'
                DriftSurf: our proposed algorithm
        """
        self.algorithms = algo_names
        self.loss = {}
        self.computation = computation.lower()
        self.rate = rate

        self.opt = getattr(Training, base_learner)

        self.dataset_name = dataset
        read_data = data.read_dataset()
        self.X, self.Y, self.n, self.d, self.drift_times = read_data.read(self.dataset_name)

        if self.dataset_name.startswith('sea'):
            self.dataset_name = 'sea'

        self.mu = hyperparameters.MU[self.dataset_name]
        self.step_size = hyperparameters.STEP_SIZE[self.dataset_name]
        self.b = hyperparameters.b[self.dataset_name] if self.dataset_name in hyperparameters.b.keys() else hyperparameters.b['default']
        self.lam = int(self.n//self.b)
        self.rho = self.lam * self.rate

    @staticmethod
    def load_dataset(dataset):
        """
            load the given dataset
        :param dataset: str
                name of the dataset to be loaded
        :return:
            features, labels, #records, #dimension, drift_times
        """

        name = dataset
        if dataset.startswith('sea'):
            name = 'sea'

        read_data = data.read_dataset()
        X, Y, n, d, drift_times = read_data.read(dataset)
        return name, X, Y, n, d, drift_times

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

    def update_loss(self, test_set, time):
        """
            computes and updates the loss of algorithms at time t over the given test set
        :param test_set:
        :param time:
        """

        for algo in self.algorithms:
            self.loss[algo][time] = getattr(self, algo).zero_one_loss(test_set)

    def setup_algorithms(self, delta, loss_fn, detector=MDDM_G(), Aware_reset = 'B', r=None, reactive_method='greedy', condition1='best_observed_perf', condition_switch='compare_trained'):
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
            if algo == 'DriftSurf': self.setup_DriftSurf(delta, loss_fn, r, reactive_method, condition1, condition_switch)
            elif algo == 'MDDM': self.setup_MDDM(detector)
            elif algo == 'Aware': self.setup_Aware(Aware_reset)
            else: getattr(self, 'setup_{0}'.format(algo))()

    def setup_DriftSurf(self, delta, loss_fn, r=None, reactive_method='greedy', condition1='best_observed_perf', condition_switch='compare_trained'):
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
        self.DriftSurf = models.LogisticRegression_DriftSurf(self.d, self.opt, delta, loss_fn, self.DriftSurf_r, reactive_method, condition1, condition_switch)

    def setup_MDDM(self, detector=MDDM_G()):
        """
            setup parameters of MDDM
        :param detector: (MDDM-A, MDDM-E, MDDM-G)
                drift detector of MDDM, defualt is set to be MDDM-D()
        """
        self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt)
        self.MDDM_drift_detector = detector

    def setup_AUE(self):
        """
            setup AUE
        """
        self.AUE = models.LogisticRegression_AUE(self.d, self.opt)

    def setup_Aware(self, reset='B'):
        """
            setup Aware
        :param reset: str (B / A)
                when to reset parameters of the predictive model in Aware: before computing loss or after - default is set to be before
        """
        self.Aware_reset = reset
        self.Aware = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S)

    def setup_SGD(self):
        """
            setup single-pass SGD
        """
        self.SGD = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.SGD)

    def setup_OBL(self):
        """
            setup oblivious algorithm
        """
        self.OBL = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.STRSAGA)

    def setup_Candor(self):
        """
            setup Candor
        """
        self.Candor = models.LogisticRegression_Candor(self.d, self.opt)

    def update_strsaga_model(self, model):
        """
            update the given model based on strsaga algorithm presented in 'Jothimurugesan, E., Tahmasbi, A., Gibbons, P., and Tirtha-pura, S. Variance-reduced stochastic gradient descent onstreaming data. InNeurIPS, pp. 9906–9915, 2018.'
        :param model:
            the model to be updated
        """
        if model:
            weight = model.get_weight() if self.computation == Training.LIMITED else 1
            lst = list(model.T_pointers)
            for s in range(int(self.rho * weight)):
                if s % 2 == 0 and lst[1] < self.S + self.lam:
                    j = lst[1]
                    lst[1] += 1
                else:
                    j = random.randrange(lst[0], lst[1])
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)
            model.update_effective_set(lst[1])

    def update_strsaga_model_biased(self, model, wp):
        if model: # limited case????
            weight = model.get_weight() if self.computation == Training.LIMITED else 1
            lst = list(model.T_pointers)
            # for s in range(int(self.rho * weight * 5)):
            for s in range(int(self.rho * weight)):
                if s % 2 == 0 and lst[1] < self.S + self.lam:
                    j = lst[1]
                    lst[1] += 1
                else:
                    j = random.randrange(lst[0], lst[1])
                point = (j, self.X[j], self.Y[j])
                model.strsaga_step_biased(point, self.step_size, self.mu, wp)
                # model.strsaga_step_biased(point, self.step_size, models.LogisticRegression_Candor.MU, wp)
            model.update_effective_set(lst[1])

    def update_sgd_model(self, model):
        """
            update the given model based on SGD algorithm
        :param model:
                the model to be updated
        """
        if model:
            weight = model.get_weight() if self.computation == Training.LIMITED else 1
            lst = list(model.T_pointers)
            for s in range(int(self.rho * weight)):

                j = random.randrange(lst[0], lst[1] + self.lam)
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)
            model.update_effective_set(lst[1] + self.lam)

    def update_sgd_model_biased(self, model, wp):
        if model:
            weight = model.get_weight() if self.computation == Training.LIMITED else 1
            lst = list(model.T_pointers)
            for s in range(int(self.rho * weight)):
                j = random.randrange(lst[0], lst[1] + self.lam)
                point = (j, self.X[j], self.Y[j])
                model.step_step_biased(point, self.step_size, models.LogisticRegression_Candor.MU, wp)
            model.update_effective_set(lst[1] + self.lam)

    # single-pass SGD
    def update_sgd_SP_model(self, model):
        """
            update the given model based on a single-pass SGD algorithm
        :param model:
                given model to be updated
        """
        if model:
            sgdOnline_T = self.S
            for s in range(min(self.lam, self.rho)):
                if sgdOnline_T < self.S + self.lam:
                    j = sgdOnline_T
                    sgdOnline_T += 1
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)

    def process_MDDM(self, time, new_batch):
        """
            MDDM's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """

        if (self.MDDM_drift_detector.test(self.MDDM, new_batch) and time != 0):
            self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S)
            self.MDDM_drift_detector.reset()
            logging.info('MDDM drift detected, reset model : {0}'.format(time))

        getattr(self, 'update_{0}_model'.format(self.opt))(self.MDDM)

    def process_AUE(self, time, new_batch):
        """
            AUE's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        self.AUE.update_weights(new_batch)
        logging.info('AUE Experts at time {0}: {1}'.format(time, [int(k / self.lam) for k in self.AUE.experts.keys()]))
        for index, expert in self.AUE.experts.items():
            getattr(self, 'update_{0}_model'.format(self.opt))(expert)

    def process_Candor(self, time, new_batch):
        """
            Candor's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        # self.Candor.update_weights_batch(new_batch)
        # for training_point in new_batch:
        #     self.Candor.update_weights(training_point)

        wp = self.Candor.get_weighted_combination()

        expert = models.LogisticRegression_expert(numpy.copy(wp), self.opt, self.S)  # alt: first arg is wp
        # expert = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S)  # alt: first arg is wp
        if time == 0:
            getattr(self, 'update_{0}_model'.format(self.opt))(expert)
        else:
            getattr(self, 'update_{0}_model_biased'.format(self.opt))(expert, wp)
        self.Candor.experts.append(expert)
        self.Candor.reset_weights()
        # for training_point in new_batch:
        #     self.Candor.update_weights(training_point)

    def process_DriftSurf(self, time, new_batch):
        """
            DriftSurf's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        self.DriftSurf.update_perf_all(new_batch, self.mu)

        if self.DriftSurf.stable:
            if self.DriftSurf.enter_reactive(self.S, new_batch, self.mu):
                self.DriftSurf_t = 0
                logging.info('DriftSurf enters reactive state : {0}'.format(time))

            else:
                # update models
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf.expert_predictive)
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf.expert_stable)

        if not self.DriftSurf.stable:
            # update models
            self.DriftSurf.update_reactive_sample_set(new_batch)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf.expert_predictive)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf.expert_reactive)

            self.DriftSurf_t += 1
            if self.DriftSurf_t == self.DriftSurf_r :
                self.DriftSurf.exit_reactive(self.S+self.lam, self.mu)

    def process_Aware(self):
        getattr(self, 'update_{0}_model'.format(self.opt))(self.Aware)

    def process_OBL(self):
        """
            oblivious algorithm's process
        """
        lst = list(self.OBL.T_pointers)
        for s in range(self.rho):
            if s % 2 == 0 and lst[1] < self.S + self.lam:
                lst[1] += 1
                j = random.randrange(lst[0], lst[1])
            point = (j, self.X[j], self.Y[j])
            self.OBL.update_step(point, self.step_size, self.mu)
        self.OBL.update_effective_set(lst[1])

    def process_SGD(self):
        self.update_sgd_SP_model(self.SGD)

    def process(self, delta=0.1, loss_fn='zero-one', drift_detectr=MDDM_G(), Aware_reset='B', r=None, reactive_method='greedy', condition1='best_observed_perf', condition_switch='compare_trained'):
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
        # self.rho = int(self.lam * rate)
        self.setup_algorithms(delta, loss_fn, drift_detectr, Aware_reset, r, reactive_method, condition1, condition_switch)

        for algo in self.algorithms:
            self.loss[algo] = [0] * self.b

        logging.info('dataset : {0}, n : {1}, b : {2}'.format(self.dataset_name, self.n, self.b))


        for time in range(self.b):

            print(time)

            if time in self.drift_times and 'Aware' in self.algorithms and self.Aware_reset == Training.BEFORE:
                self.setup_Aware()

            # measure accuracy over upcoming batch
            test_set = [(i, self.X[i], self.Y[i]) for i in range(self.S, self.S + self.lam)]
            self.update_loss(test_set, time)

            # if time in self.drift_times and 'Aware' in self.algorithms and self.Aware_reset == Training.AFTER:
            #     self.setup_Aware()

            for algo in self.algorithms:
                if algo in ['SGD', 'OBL', 'Aware']: getattr(self, 'process_{0}'.format(algo))()
                else: getattr(self, 'process_{0}'.format(algo))(time, test_set)


            self.S += self.lam
        return self.loss

