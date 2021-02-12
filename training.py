import models
import read_data as data
import read_data_noDrift as data_noDrift
import hyperparameters
from drift_detection.__init__ import *
import random
import numpy
import logging

numpy.random.seed(1)
random.seed(1)

class Training:

    STRSAGA = models.Opt.STRSAGA
    SGD = models.Opt.SGD
    LIMITED = 'all_models'
    UNLIMITED = 'each_model'
    BEFORE = 'B'
    AFTER = 'A'

    def __init__(self, dataset, computation='each_model', rate=2, opt = models.Opt.STRSAGA.upper(), algo_names=['Aware', 'SGD', 'OBL', 'MDDM', 'AUE', 'DriftSurf'], drift=True, base_learner=models.DriftSurf_v1.LR):
        """

        :param dataset: str
                name of the dataset
        :param computation: str (algorithm / model)
                algorithm: rho computational power is given to each algorithm and needs to be divided between learners
                model: rho computational power is given to each model in each algorithm
        :param rate: int
                rate = (rho/lam)
        :param opt: (STRSAGA / SGD)
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
        self.time_drift_detected = {}
        self.FPR = {}
        self.avg_RT = {}

        self.computation = computation.lower()
        self.rate = rate
        self.base_learner = base_learner
        self.opt = getattr(Training, opt)

        self.dataset_name = dataset
        if drift: read_data = data.read_dataset()
        else: read_data = data_noDrift.read_dataset_noDrift()
        self.X, self.Y, self.n, self.d, self.drift_times = read_data.read(self.dataset_name)

        if self.base_learner != 'LR':
            self.change_datasets_labels()
        models.DatasetName.name = self.dataset_name

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

    def change_datasets_labels(self):
        # convert into numpy arrays, and relabel 1/-1 to 1/0
        X0 = []
        for xx in self.X:
            xx0 = numpy.zeros((1, self.d))
            if not self.dataset_name.startswith('airline'):
                for (k, v) in xx.items():
                    xx0[0][k] = v
            elif self.dataset_name == 'airline_unprocessed':
                for k in range(self.d):
                    xx0[0][k] = xx[k]
            X0.append(xx0)
        self.X = X0

        Y0 = []
        for yy in self.Y:
            yy0 = numpy.zeros(1)
            yy0[0] = 1 if yy == 1 else 0
            Y0.append(yy0)
        self.Y = Y0

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

    def setup_algorithms(self, delta, loss_fn, detector=MDDM_G(), Aware_reset = 'B', r=None, reactive_method=models.DriftSurf_v1.GREEDY):
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
            if algo.startswith('DriftSurf'): getattr(self, 'setup_{0}'.format(algo))(delta, loss_fn, r, reactive_method, self.base_learner)
            elif algo == 'Standard': self.setup_Standard(delta, loss_fn)
            elif algo == 'MDDM': self.setup_MDDM(detector)
            elif algo == 'Aware': self.setup_Aware(Aware_reset)
            else: getattr(self, 'setup_{0}'.format(algo))()

    def setup_DriftSurf_v1(self, delta, loss_fn, r=None, method=models.DriftSurf_v1.GREEDY, base_learner=models.DriftSurf_v1.LR):
        """
            setup parameters of DriftSurf
        :param delta: float
                delta-degration in performance is considered as a sign of drift
        :param loss_fn: str (reg, zero-one)
                loss function that DriftSurf check for performance degration
        :param r: int
                length of the reactive state
        """

        self.DriftSurf_v1_t = 0
        self.DriftSurf_v1_r = r if r else (hyperparameters.r['default'] if base_learner== models.DriftSurf_v1.LR else hyperparameters.r['others'])
        self.DriftSurf_v1 = models.DriftSurf_v1(self.d, self.opt, delta, loss_fn, method, base_learner)

    def setup_Standard(self, delta, loss_fn):
        """
            setup parameters of DriftSurf
        :param delta: float
                delta-degration in performance is considered as a sign of drift
        :param loss_fn: str (reg, zero-one)
                loss function that DriftSurf check for performance degration
        :param r: int
                length of the reactive state
        """
        self.Standard = models.Standard(self.d, self.opt, delta, loss_fn, self.base_learner)

    def setup_DriftSurf_v2(self, delta, loss_fn, r=None, method=models.DriftSurf_v2.GREEDY, base_learner=models.DriftSurf_v2.LR):
        """
            setup parameters of DriftSurf
        :param delta: float
                delta-degration in performance is considered as a sign of drift
        :param loss_fn: str (reg, zero-one)
                loss function that DriftSurf check for performance degration
        :param r: int
                length of the reactive state
        """

        self.DriftSurf_v2_t = 0
        self.DriftSurf_v2_r = r if r else (hyperparameters.r['default'] if base_learner==models.DriftSurf_v2.LR else hyperparameters.r['others'])
        self.DriftSurf_v2 = models.DriftSurf_v2(self.d, self.opt, delta, loss_fn, method, base_learner)

    def setup_DriftSurf_v3(self, delta, loss_fn, r=None, method=models.DriftSurf_v2.GREEDY, base_learner=models.DriftSurf_v2.LR):
        """
            setup parameters of DriftSurf
        :param delta: float
                delta-degration in performance is considered as a sign of drift
        :param loss_fn: str (reg, zero-one)
                loss function that DriftSurf check for performance degration
        :param r: int
                length of the reactive state
        """

        self.DriftSurf_v3_t = 0
        self.DriftSurf_v3_r = r if r else (hyperparameters.r['default'] if base_learner==models.DriftSurf_v2.LR else hyperparameters.r['others'])
        self.DriftSurf_v3 = models.DriftSurf_v2(self.d, self.opt, delta, loss_fn, method, base_learner)

    def setup_MDDM(self, detector=MDDM_G()):
        """
            setup parameters of MDDM
        :param detector: (MDDM-A, MDDM-E, MDDM-G)
                drift detector of MDDM, defualt is set to be MDDM-D()
        """
        self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner, self.opt)
        self.MDDM_drift_detector = detector

    def setup_AUE(self):
        """
            setup AUE
        """
        # if self.base_learner == models.DriftSurf_v1.LR:  self.AUE = models.LogisticRegression_AUE(self.d, self.opt)
        self.AUE = models.AUE(self.d, self.opt, self.base_learner)

    def setup_Aware(self, reset='B'):
        """
            setup Aware
        :param reset: str (B / A)
                when to reset parameters of the predictive model in Aware: before computing loss or after - default is set to be before
        """
        self.Aware_reset = reset
        self.Aware = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner,self.opt, self.S)

    def setup_SGD(self):
        """
            setup single-pass SGD
        """
        self.SGD = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.SGD) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner, Training.SGD)

    def setup_OBL(self):
        """
            setup oblivious algorithm
        """
        self.OBL = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.STRSAGA) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner, Training.STRSAGA)

    def setup_Candor(self):
        """
            setup Candor
        """
        self.Candor = models.LogisticRegression_Candor(self.d, self.opt) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner, self.opt)
        
    def setup_HAT(self):
        """
            setup HAT
        """
        self.HAT = models.HoeffdingAdaptiveTree()
        
    def update_model(self, model):        
        for j in range(self.S, self.S + self.lam):
            point = (j, self.X[j], self.Y[j])
            model.update_step(point, self.step_size, self.mu)
        model.update_effective_set(self.S + self.lam)

    def update_strsaga_model(self, model):
        """
            update the given model based on strsaga algorithm presented in 'Jothimurugesan, E., Tahmasbi, A., Gibbons, P., and Tirtha-pura, S. Variance-reduced stochastic gradient descent onstreaming data. InNeurIPS, pp. 9906–9915, 2018.'
        :param model:
            the model to be updated
        """
        if self.base_learner != 'LR':
            if model: self.update_model(model)
        else:
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
        if self.base_learner != 'LR':
            if model: self.update_model(model)
        else:
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
                    model.strsaga_step_biased(point, self.step_size, self.mu, wp)
                model.update_effective_set(lst[1])

    def update_sgd_model(self, model):
        """
            update the given model based on SGD algorithm
        :param model:
                the model to be updated
        """
        # if model: self.update_model(model)
        if model:
            weight = model.get_weight() if self.computation == Training.LIMITED else 1
            lst = list(model.T_pointers)
            total_iters = int(self.rho * weight)
            for j in range(self.S, self.S + min(self.lam, total_iters)):
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)
            model.update_effective_set(lst[1] + self.lam)
            for s in range(total_iters - self.lam):
                j = random.randrange(lst[0], lst[1] + self.lam)
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)

    def update_sgd_model_biased(self, model, wp):
        if self.base_learner != 'LR':
            if model: self.update_model(model)
        else:
            if model:
                weight = model.get_weight() if self.computation == Training.LIMITED else 1
                lst = list(model.T_pointers)
                for s in range(int(self.rho * weight)):
                    j = random.randrange(lst[0], lst[1] + self.lam)
                    point = (j, self.X[j], self.Y[j])
                    model.step_step_biased(point, self.step_size, models.Candor.MU, wp)
                model.update_effective_set(lst[1] + self.lam)

    # single-pass SGD
    def update_sgd_SP_model(self, model):
        """
            update the given model based on a single-pass SGD algorithm
        :param model:
                given model to be updated
        """
        if self.base_learner != 'LR':
            if model: self.update_model(model)
        else:
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
            self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner, self.opt, self.S)
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
        # update_all = False

        wp = self.Candor.get_weighted_combination()
        expert = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S) if self.base_learner == 'LR' else models.OtherBaseLearners_expert(self.base_learner, self.opt, self.S) # alt: first arg is wp
        if time == 0:
            getattr(self, 'update_{0}_model'.format(self.opt))(expert)
        else:
            getattr(self, 'update_{0}_model_biased'.format(self.opt))(expert, wp)

        self.Candor.experts.append((expert, wp))
        self.Candor.reset_weights()
        
    def process_HAT(self, time, new_batch):
        for j in range(self.S, self.S + self.lam):
            point = (j, self.X[j], self.Y[j])
            self.HAT.update_model(point)

    def process_DriftSurf_v1(self, time, new_batch):
        """
            DriftSurf's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        logging.info('DriftSurf_v1:')
        self.DriftSurf_v1.update_perf_all(new_batch, self.mu)

        if self.DriftSurf_v1.stable:
            if self.DriftSurf_v1.enter_reactive(self.S, new_batch, self.mu):
                self.DriftSurf_v1_t = 0
                logging.info('enters reactive state : {0}'.format(time))

            else:
                # update models
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v1.expert_predictive)
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v1.expert_stable)

        if not self.DriftSurf_v1.stable:
            # update models
            self.DriftSurf_v1.update_reactive_sample_set(new_batch)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v1.expert_predictive)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v1.expert_reactive)

            self.DriftSurf_v1_t += 1
            if self.DriftSurf_v1_t == self.DriftSurf_v1_r :
                self.DriftSurf_v1.exit_reactive(self.S+self.lam, self.mu)

    def process_DriftSurf_v2_Reactive(self, time, new_batch):

        if self.DriftSurf_v2.enter_reactive_condition():
            if self.DriftSurf_v2_reEnter:
                self.DriftSurf_v2.enter_reactive(self.S, new_batch, self.mu)
                self.DriftSurf_v2_t = 0
                self.DriftSurf_v2_reEnter = False
                logging.info('enters reactive state again {}'.format(time))

        else:
            self.DriftSurf_v2_reEnter = True

        self.DriftSurf_v2_t += 1

        if self.DriftSurf_v2_r / 2 < self.DriftSurf_v2_t <= self.DriftSurf_v2_r :
            self.DriftSurf_v2.update_reactive_sample_set(new_batch)

        # update models
        getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v2.expert_predictive)
        getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v2.expert_reactive)

        if self.DriftSurf_v2_t == self.DriftSurf_v2_r / 2:
            self.DriftSurf_v2.frozen_reactive_model()

        elif self.DriftSurf_v2_t == self.DriftSurf_v2_r :
            drift_detected, num = self.DriftSurf_v2.exit_reactive(self.S + self.lam, self.mu)
            self.time_drift_detected['DriftSurf_v2'].append(time + num)

    def process_DriftSurf_v2(self, time, new_batch):
        """
            DriftSurf's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        logging.info('DriftSurf_v2: ')
        self.DriftSurf_v2.update_perf_all(new_batch, self.mu)

        if self.DriftSurf_v2.stable:
            if self.DriftSurf_v2.enter_reactive(self.S, new_batch, self.mu):
                self.DriftSurf_v2_t = 0
                self.DriftSurf_v2_reEnter = False
                logging.info('enters reactive state : {0}'.format(time))

            else:
                # update models
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v2.expert_predictive)

        if not self.DriftSurf_v2.stable:

            self.process_DriftSurf_v2_Reactive(time, new_batch)

    def process_DriftSurf_v3_Reactive(self, time, new_batch):

        if not self.DriftSurf_v3.enter_reactive_condition():
            self.DriftSurf_v3.stable = True
            logging.info('exists reactive state early at time {}'.format(time))
            self.process_DriftSurf_v3_stable(time, new_batch)
        else:
            self.DriftSurf_v3_t += 1
            # update models
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v3.expert_predictive)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v3.expert_reactive)

            if self.DriftSurf_v3_r / 2 < self.DriftSurf_v3_t <= self.DriftSurf_v3_r :
                self.DriftSurf_v3.update_reactive_sample_set(new_batch)


            if self.DriftSurf_v3_t == self.DriftSurf_v3_r / 2:
                self.DriftSurf_v3.frozen_reactive_model()

            elif self.DriftSurf_v3_t == self.DriftSurf_v3_r :
                drift_detected, num = self.DriftSurf_v3.exit_reactive(self.S + self.lam, self.mu)
                self.time_drift_detected['DriftSurf_v3'].append(time + num)

    def process_DriftSurf_v3_stable(self, time, new_batch):

        if self.DriftSurf_v3.stable:
            if self.DriftSurf_v3.enter_reactive(self.S, new_batch, self.mu):
                self.DriftSurf_v3_t = 0
                logging.info('enters reactive state : {0}'.format(time))

            else:
                # update models
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DriftSurf_v3.expert_predictive)

    def process_DriftSurf_v3(self, time, new_batch):
        """
            DriftSurf's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        logging.info("DriftSurf_v3: ")
        self.DriftSurf_v3.update_perf_all(new_batch, self.mu)

        if self.DriftSurf_v3.stable:
            self.process_DriftSurf_v3_stable(time, new_batch)
        if not self.DriftSurf_v3.stable:
            self.process_DriftSurf_v3_Reactive(time, new_batch)

    def process_Standard(self, time, new_batch):
        """
            DriftSurf's process at time t given a newly arrived batch of data points
        :param time: int
                time step
        :param new_batch:
                newly arrived batch of data points
        """
        self.Standard.update_perf_all(new_batch, self.mu)

        if self.Standard.detect_drift():
            logging.info('Standard detects a drift : {0}'.format(time))
            self.Standard.switch_model(self.S)

        # update models
        getattr(self, 'update_{0}_model'.format(self.opt))(self.Standard.expert_predictive)

    def process_Aware(self):
        getattr(self, 'update_{0}_model'.format(self.opt))(self.Aware)

    def process_OBL(self):
        """
            oblivious algorithm's process
        """
        if self.base_learner != 'LR':
            self.update_model(self.OBL)
        else:
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

    def calc_FPR(self):

        for algo in self.algorithms:

            self.drift_times.append(self.b)
            false_detection = []
            true_detection = []
            recovery_time = []
            i = 0

            for j in range(len(self.drift_times)-1):
                while i < len(self.time_drift_detected[algo]) and self.time_drift_detected[algo][i] < self.drift_times[j]:
                    false_detection.append(self.time_drift_detected[algo][i])
                    i += 1
                if i < len(self.time_drift_detected[algo]) and self.drift_times[j+1] > self.time_drift_detected[algo][i] >= self.drift_times[j]:
                    true_detection.append(self.time_drift_detected[algo][i])
                    recovery_time.append(self.time_drift_detected[algo][i] - self.drift_times[j])
                    i += 1
                else:
                    recovery_time.append(self.drift_times[j+1] - self.drift_times[j])

            while i < len(self.time_drift_detected[algo]):
                false_detection.append(self.time_drift_detected[algo][i])
                i += 1
            print(algo, true_detection, false_detection)
        # for algo in self.algorithms:
        #
        #     self.drift_times.append(self.b)
        #
        #     false_detection = []
        #     true_detection = []
        #     recovery_time = []
        #     i = 0
        #     if len(self.time_drift_detected[algo]) == 0:
        #         for j in range(len(self.drift_times) - 2):
        #             recovery_time.append(self.drift_times[j+1]- self.drift_times[j] )
        #
        #     for time in self.time_drift_detected[algo]:
        #
        #         if time < self.drift_times[i]:
        #             false_detection.append(time)
        #         else:
        #             while i < len(self.drift_times) - 1 and not (time < self.drift_times[i+1]):
        #                 recovery_time.append(self.drift_times[i+1] - self.drift_times[i])
        #                 i += 1
        #
        #             true_detection.append(time)
        #             recovery_time.append(time - self.drift_times[i])
        #             i += 1
            del self.drift_times[-1]
            self.FPR[algo] = 0 if len(self.time_drift_detected[algo]) == 0 else round(len(false_detection) / len(self.time_drift_detected[algo]), 2)
            self.avg_RT[algo] = 0 if len(self.drift_times) == 0 else round(sum(recovery_time)/len(self.drift_times), 2)


    def process(self, delta=0.1, loss_fn='reg', drift_detectr=MDDM_G(), Aware_reset='B', r=None, reactive_method=models.DriftSurf_v1.GREEDY):
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
        self.setup_algorithms(delta, loss_fn, drift_detectr, Aware_reset, r, reactive_method)

        for algo in self.algorithms:
            self.loss[algo] = [0] * self.b
            self.time_drift_detected[algo] = []

        logging.info('dataset : {0}, n : {1}, b : {2}'.format(self.dataset_name, self.n, self.b))

        for time in range(self.b):

            print(time)
            logging.info(time)

            if time in self.drift_times and 'Aware' in self.algorithms and self.Aware_reset == Training.BEFORE:
                self.setup_Aware()

            # measure accuracy over upcoming batch
            test_set = [(i, self.X[i], self.Y[i]) for i in range(self.S, self.S + self.lam)]
            self.update_loss(test_set, time)

            if time in self.drift_times and 'Aware' in self.algorithms and self.Aware_reset == Training.AFTER:
                self.setup_Aware()

            for algo in self.algorithms:
                if algo in ['SGD', 'OBL', 'Aware']: getattr(self, 'process_{0}'.format(algo))()
                else: getattr(self, 'process_{0}'.format(algo))(time, test_set)

            self.S += self.lam

        self.calc_FPR()
        print('FPR : {0} and RT : {1}' .format(self.FPR, self.avg_RT))
        logging.info('FPR : {0} and RT : {1}' .format(self.FPR, self.avg_RT))

        print(self.time_drift_detected, self.drift_times)
        logging.info('actual time of drift: {0}, model changed at times: {1}'.format(self.drift_times, self.time_drift_detected))

        return self.loss, self.FPR, self.avg_RT