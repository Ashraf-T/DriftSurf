from math import e
import math
import os
from random import seed
from random import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# H(n) = c * n^(-\alpha)
# let c = 1 and \alpha = 0.5


class numericalAnalyis_stationary():

    def __init__(self, num_batch, size_batch, num_iter, delta, max_T):
        self.b = num_batch
        self.m = size_batch
        self.n = [ele * self.b for ele in self.m]

        self.iter = num_iter
        self.delta = delta
        self.T = list(range(1, max_T))

        self.ave_age = []
        self.ave_loss = []

    def enter_reactive_1(self, m, age):

        value = random()
        return value < (1 + 1. / self.delta) * (1. / (age * m) ** .25)


    def enter_reactive_2(self, m, age_b):
        value = random()
        return value < (e ** -(age_b * m))

    def eneter_reactive(self, batch_size, age, age_b, entered_reactive):
        return self.enter_reactive_1(batch_size, age) or (entered_reactive and self.enter_reactive_2(batch_size, age_b))

    def reactive_state(self, time, age, t, sum, num_experts):
        time += t
        if self.switch_to_new(age, t):

            sum += age
            age = t
            num_experts += 1

        else:

            age += t

        return True, time, sum, age, num_experts, age

    def switch_to_new(self, age, t):
        value = random()
        return value < min(1, e ** (-(age - t) / (3 * t)))

    def compute_ave_age(self):

        for batch_size in self.m:

            age_for_m = []
            for t in self.T:

                sum_total = 0

                for i in range(self.iter):

                    time = 1
                    age = 1
                    num_experts = 1
                    entered_reactive = False
                    age_b = 0
                    sum = 0

                    while time <= self.b:

                        if self.eneter_reactive(batch_size, age, age_b, entered_reactive):
                            entered_reactive, time, sum, age, num_experts, age_b = self.reactive_state(time, age, t, sum, num_experts)

                        age += 1
                        time += 1

                    sum += min(b, age)
                    ave = sum / num_experts

                    sum_total += ave

                age_for_m.append(sum_total / self.iter)
            self.ave_age.append(age_for_m)

    def compute_ave_loss(self):

        # self.compute_ave_age()
        for i in range(len(self.ave_age)):
            risk = []
            for j in range(len(self.ave_age[i])):
                risk.append(1. / math.sqrt(self.ave_age[i][j]))
            self.ave_loss.append(risk)


    def average_age(self):
        return self.ave_age

    def plot(self):

        current_time = time.strftime('%Y-%m-%d_%H%M%S')
        path = os.path.join('output', 'stationary', current_time)
        os.makedirs(path)

        plt.figure(1)
        plt.clf()
        # plt.plot(self.T, self.T, '--')
        for i in range(len(self.ave_age)):
            plt.plot(self.T, self.ave_age[i], lw=1, label="n_1e{}".format(int(math.log10(self.m[i]*100))))
        # plt.yscale('log')
        plt.ylabel('average age')
        plt.xlabel('T')
        plt.title('no drift')
        plt.legend()
        # plt.savefig(os.path.join('ave_age.eps'), format='eps')
        plt.savefig(os.path.join(path, 'ave_age.png'), format='png', dpi=200)

        plt.figure(2)
        plt.clf()
        # print(ave_age)
        for i in range(len(self.ave_loss)):
            plt.plot(self.T, self.ave_loss[i], lw=1, label="n_1e{}".format(int(math.log10(self.m[i]*100))))
        # plt.yscale('log')
        plt.ylabel('average loss')
        plt.xlabel('T')
        plt.title('no drift')
        plt.legend()
        plt.savefig(os.path.join(path, 'ave_loss.png'), format='png', dpi=200)
        # plt.show()


class numericalAnalysis_drift():

    def __init__(self, num_batch, size_batch, delta, max_T, ave_age_stationary, factor, approximation):
        self.b = num_batch
        self.m = size_batch
        self.n = [ele * self.b for ele in self.m]

        self.delta = delta
        self.T = list(range(1, max_T))
        self.drift_rate = [0.2, 0.4, 0.6, 0.8, 1]
        self.approx = approximation
        self.ave_age = ave_age_stationary
        self.factor = factor

        self.p_normal = []
        self.p_linear = []

        self.p_switch = []
        self.rec_time = []

    def p_enterReactive_normal(self, batch_size, t, drift_rate):

        batch_index = self.m.index(batch_size)
        drift_index = self.drift_rate.index(drift_rate)
        t_index = self.T.index(t)

        return self.p_normal[batch_index][drift_index][t_index]


    def compute_p_enterReactive_normal(self):

        for i in range(len(self.m)):

            p_d = []
            for drift in self.drift_rate:

                p_t = []
                for j in range(len(self.T)):

                    p = min(1, 1. / (math.sqrt(self.ave_age[i][j] * self.factor * self.m[i]))) # loss before drift
                    q = min(1, p + drift) # loss after drift

                    s_square = p * (1 - p) + q * (1 - q)

                    temp = math.sqrt(self.m[i]) * (drift - self.delta) / math.sqrt(s_square)
                    p_t.append(norm.cdf(temp))

                p_d.append(p_t)

            self.p_normal.append(p_d)


    def p_enterReactive_linear(self, drif_rate):

        return self.p_linear[self.drift_rate.index(drif_rate)]

    def compute_p_enterReactive_linear(self):

         for drift in self.drift_rate:
            self.p_linear.append(drift)


    def p_switch_drift(self, t, drift_rate, batch_size):

        batch_index = self.m.index(batch_size)
        t_index = self.T.index(t)
        drift_index = self.drift_rate.index(drift_rate)

        return self.p_switch[batch_index][drift_index][t_index]


    def compute_p_switch(self):

        for m in self.m:
            p_switch_temp1 = []
            for d in range(len(self.drift_rate)):
                p_switch_temp2 = []
                for t in self.T:
                    age = self.ave_age[self.m.index(m)][self.T.index(t)] * self.factor
                    p_switch_temp2.append(min(1, e ** (-((age - t) / (3 * t)) ** (1 - self.drift_rate[d]) + (
                                self.drift_rate[d] * ((m) ** 0.5 / (age)) ** (.75)))))
                p_switch_temp1.append(p_switch_temp2)
            self.p_switch.append(p_switch_temp1)


    def compute_p_switch_1(self):

        for m in self.m:
            p_switch_temp1 = []
            for d in range(len(self.drift_rate)):
                p_switch_temp2 = []
                for t in self.T:
                    age = self.ave_age[self.m.index(m)][self.T.index(t)] * self.factor
                    p = 1. / math.sqrt(age * m) + self.drift_rate[d]
                    q = 1. / math.sqrt(t * m)
                    s_square = p * (1 - p) + q * (1 - q)
                    p_switch_temp2.append(norm.cdf((p - q) * math.sqrt(m * t / s_square)))
                p_switch_temp1.append(p_switch_temp2)
            self.p_switch.append(p_switch_temp1)


    def compute_p_switch_2(self, epsilon):

        for m in self.m:
            p_switch_temp1 = []
            for d in range(len(self.drift_rate)):
                p_switch_temp2 = []
                for t in self.T:
                    age = self.ave_age[self.m.index(m)][self.T.index(t)] * self.factor
                    p = 1. / math.sqrt(age * m) + self.drift_rate[d]
                    q = 1. / math.sqrt(t * m)
                    s_square = p * (1 - p) + q * (1 - q)
                    p_switch_temp2.append(norm.cdf(((p - q) * m * t - epsilon) / math.sqrt(m * t * s_square)))
                p_switch_temp1.append(p_switch_temp2)
            self.p_switch.append(p_switch_temp1)


    def compute_p_switch_3(self):

        for m in self.m:
            p_switch_temp1 = []
            for d in range(len(self.drift_rate)):
                p_switch_temp2 = []
                for t in self.T:
                    age = self.ave_age[self.m.index(m)][self.T.index(t)] * self.factor
                    p = 1. / math.sqrt(age * m) + self.drift_rate[d]
                    q = (1./t) * sum(1./math.sqrt(tt * m) for tt in range(1,t+1))
                    s_square = p * (1 - p) + (1./t) * sum((1. / math.sqrt(tt * m))*(1 - 1. / math.sqrt(tt * m)) for tt in range(1,t+1))

                    p_switch_temp2.append(norm.cdf((p - q) * math.sqrt(m * t / s_square)))
                p_switch_temp1.append(p_switch_temp2)
            self.p_switch.append(p_switch_temp1)


    def recovery_time_formula(self, epsilon_1, epsilon_2, t, drift_rate, batch_size):

        p_switch = self.p_switch_drift(t, drift_rate, batch_size)
        p_enterReactive = self.p_enterReactive_linear(drift_rate) if self.approx == 'linear' else self.p_enterReactive_normal(batch_size, t, drift_rate)

        k = max(1, math.sqrt((1-epsilon_1)/epsilon_1) * ((1 - p_switch)/p_switch**2))

        return min(50, k * t + (2./p_enterReactive)*(math.log(1./epsilon_2) + k * math.log(2)))


    def compute_recovery_time(self, epsilon_1, epsilon_2):

        for i in range(len(self.m)):
            recovery_time_temp1 = []
            for d in range(len(self.drift_rate)):
                recovery_time_temp2 = []
                for t in range(len(self.T)):
                    recovery_time_temp2.append(self.recovery_time_formula(epsilon_1, epsilon_2, self.T[t], self.drift_rate[d], self.m[i]))
                recovery_time_temp1.append(recovery_time_temp2)
            self.rec_time.append(recovery_time_temp1)

    def plot(self):

        current_time = time.strftime('%Y-%m-%d_%H%M%S')
        path = os.path.join('output', 'drift', 'p_switch', current_time)
        os.makedirs(path)

        for i in range(len(self.n)):
            plt.figure(i)
            plt.clf()
            for d in range(len(self.drift_rate)):
                plt.plot(self.T, self.p_switch[i][d], lw=1, label="d_{}".format(str(round(self.drift_rate[d], 2))))
            # plt.xscale('log')
            plt.ylabel('p*: probability of switching-after drift')
            plt.ylim(-0.1,1.1)
            plt.xlabel('T')
            plt.title('drift - n=1e{0}'.format(int(math.log10(self.n[i]))))
            plt.legend()
            plt.savefig(os.path.join(path, 'p_switch, n=1e{0}.png'.format(int(math.log10(self.n[i])))), format='png', dpi=200)
            # plt.show()

        if self.approx == 'linear':
            path = os.path.join('output', 'drift', 'p-linear', current_time)
            os.makedirs(path)

            plt.figure(1)
            plt.clf()
            for d in range(len(self.drift_rate)):
                plt.plot(self.T, [self.p_linear[d]] * len(self.T), lw=1,
                         label="d_{}".format(str(round(self.drift_rate[d], 2))))
            # plt.xscale('log')
            plt.ylabel('p: probability of entering reactive state-linear')
            plt.ylim(-0.1, 1.1)
            plt.xlabel('T')
            plt.title('p _ linear')
            plt.legend()
            plt.savefig(os.path.join(path, 'p.png'), format='png', dpi=200)
            # plt.show()

        else:
            path = os.path.join('output', 'drift', 'p-normal', current_time)
            os.makedirs(path)

            for i in range(len(self.n)):
                plt.figure(i)
                plt.clf()
                for d in range(len(self.drift_rate)):
                    plt.plot(self.T, self.p_normal[i][d], lw=1, label="d_{}".format(str(round(self.drift_rate[d], 2))))
                # plt.xscale('log')
                plt.ylabel('p: probability of entering reactive state-normal')
                plt.ylim(-0.1,1.1)
                plt.xlabel('T')
                plt.title('drift - n=1e{0}'.format(int(math.log10(self.n[i]))))
                plt.legend()
                plt.savefig(os.path.join(path, 'n=1e{0}.png'.format(int(math.log10(self.n[i])))), format='png', dpi=200)
                # plt.show()


        path = os.path.join('output', 'drift', 'rec_time', self.approx, current_time)
        os.makedirs(path)

        for i in range(len(self.n)):
            plt.figure(i)
            plt.clf()
            for d in range(len(self.drift_rate)):
                plt.plot(self.T, self.rec_time[i][d], lw=1, label="d_{}".format(str(round(self.drift_rate[d], 2))))
                # plt.xscale('log')
            plt.ylabel('recovery_time')
            plt.xlabel('T')
            plt.title('drift - n=1e{0}'.format(int(math.log10(self.n[i]))))
            plt.legend()
            plt.savefig(os.path.join(path, 'recoveryTime, {0}, n=1e{1}.png'.format(self.approx, int(math.log10(self.n[i])))), format='png', dpi=200)
            # plt.show()



class numericalAnalysis():

    def __init__(self, num_batch, size_batch, num_iter, delta, max_T, approximation):
        self.b = num_batch
        self.drift_time = self.b//2

        self.iter = num_iter
        self.m = size_batch
        self.n = [ele * self.b for ele in self.m]

        self.delta = delta
        self.T = list(range(1, max_T))
        self.drift_rate = [0.2, 0.4, 0.6, 0.8, 1]

        self.approx = approximation
        self.rec_time = []

        self.p_normal = []
        self.p_linear = []
        self.p_switch = []

    def enter_reactive_1(self, m, age):

        value = random()
        return value < (1 + 1. / self.delta) * (1. / (age * m) ** .25)


    def enter_reactive_2(self, m, age_b):
        value = random()
        return value < (e ** -(age_b * m))

    def eneter_reactive_stationary(self, batch_size, age, age_b, entered_reactive):
        return self.enter_reactive_1(batch_size, age) or (entered_reactive and self.enter_reactive_2(batch_size, age_b))

    def reactive_state_stationary(self, age, t):

        if self.switch_to_new_stationary(age, t):

            age = t

        else:

            age += t

        return age

    def switch_to_new_stationary(self, age, t):
        value = random()
        return value < min(1, e ** (-(age - t) / (3 * t)))


    def enter_reactive_drift(self, drift_rate, age, batch_size):

        value = random()
        if self.approx == 'linear':
            return value < drift_rate

        else:
            p = min(1, 1. / (math.sqrt(age * batch_size))) #loss before drift
            q = min(1, p + drift_rate) #loss after drift

            s_square = p * (1 - p) + q * (1 - q)
            temp = math.sqrt(batch_size) * (drift_rate - self.delta) / math.sqrt(s_square)

            return value < norm.cdf(temp)

    def switch_to_new_dirft(self, age, t, drift_rate, batch_size):

        value = random()
        return value < min(1, e ** (-((age - t) / (3 * t)) ** (1 - drift_rate) + (
                drift_rate * ((batch_size) ** 0.5 / (age)) ** (.75))))

    def process(self):

        for batch_size in self.m:

            recovery_time_for_m = []

            for drift in self.drift_rate:

                recovery_time_for_d = []
                for t in self.T:

                    sum = 0

                    for i in range(self.iter):

                        time = 1
                        age = 1
                        entered_reactive = False
                        age_b = 0

                        while time < self.drift_time:

                            if self.eneter_reactive_stationary(batch_size, age, age_b, entered_reactive):
                                entered_reactive = True
                                time += t
                                age = self.reactive_state_stationary(age, t)
                                age_b = age

                            age += 1
                            time += 1

                        if time >= self.drift_time :
                            recovery_time = 0
                            recovered = False

                        while time <= self.b and not recovered:

                            recovery_time += 1
                            if self.enter_reactive_drift(drift, age, batch_size):

                                time += t
                                recovery_time += t

                                if self.switch_to_new_dirft(age, t, drift, batch_size):
                                    recovered = True
                                    age = t
                                else:
                                    age += t

                            age += 1
                            time += 1

                        sum += recovery_time

                    recovery_time_for_d.append(sum / self.iter)
                recovery_time_for_m.append(recovery_time_for_d)
            self.rec_time.append(recovery_time_for_m)


    def plot(self):

        current_time = time.strftime('%Y-%m-%d_%H%M%S')
        path = os.path.join('output', 'stationary-drift', 'rec_time', self.approx, current_time)
        os.makedirs(path)

        for i in range(len(self.n)):
            plt.figure(i)
            plt.clf()
            for d in range(len(self.drift_rate)):
                plt.plot(self.T, self.rec_time[i][d], lw=1, label="d_{}".format(str(round(self.drift_rate[d], 2))))
                # plt.xscale('log')
            plt.ylabel('recovery_time')
            plt.xlabel('T')
            plt.title('drift - n=1e{0}'.format(int(math.log10(self.n[i]))))
            plt.legend()
            plt.savefig(os.path.join(path, 'recoverTime, {0}, n=1e{1}.png'.format(self.approx, int(math.log10(self.n[i])))), format='png', dpi=200)
            # plt.show()



if __name__ == '__main__':

    seed(1)

    b = 100
    mm = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    num = int(1e3)
    delta_s = 0.1
    max_T = 12

    num_analysis1 = numericalAnalyis_stationary(b, mm, num, delta_s, max_T)
    num_analysis1.compute_ave_age()
    num_analysis1.compute_ave_loss()
    num_analysis1.plot()

    ave_age_stationary = num_analysis1.average_age()
    factor = 1
    epsilon_1 = 0.1
    epsilon_2 = 0.1
    epsilon = 5e3
    approximation = 'linear'

    num_analysis2 = numericalAnalysis_drift(b, mm, delta_s, max_T, ave_age_stationary, factor, approximation)
    num_analysis2.compute_p_enterReactive_linear()
    num_analysis2.compute_p_enterReactive_normal()
    num_analysis2.compute_p_switch_1()
    # num_analysis2.compute_p_switch_2(epsilon)
    # num_analysis2.compute_p_switch_3()
    num_analysis2.compute_recovery_time(epsilon_1, epsilon_2)
    num_analysis2.plot()

    num_analysis3 = numericalAnalysis(b, mm, num, delta_s, max_T, approximation)
    num_analysis3.process()
    num_analysis3.plot()









