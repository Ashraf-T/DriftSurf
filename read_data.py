import csv
import math

class read_dataset:

    def read(self, dataset_name):

        if dataset_name.startswith('sea'):
            name = 'sea'
            noise_level = dataset_name[3:]
            return getattr(self, name)(self,noise_level)

        elif dataset_name.startswith('hyperplane'):
            name = 'hyperplane'
            type = dataset_name[11:]
            return getattr(self, name)(self,type)

        else:
            return getattr(self, dataset_name)(self)

    # synthetic datasets :
    def sine1(self):

        n = 10000
        d = 2 + 1
        drift_times = [20,40,60,80]

        X = []
        Y = []

        with open('data/synthetic/sine1.arff') as file:
            i = 0

            for line in file:
                if i > 5:
                    fields = line.strip().split(',')
                    label = 1 if (fields[len(fields) - 1]) == 'p' else -1
                    features = {0: 1}
                    for j in range(len(fields) - 1):
                        features[j + 1] = float(fields[j])

                    X.append(features)
                    Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def sea(self, noise_level=10):

        n = 100000
        d = 3 + 1
        drift_times = [25, 50, 75]

        X = []
        Y = []

        with open('data/synthetic/sea_abrupt{0}.arff'.format(noise_level)) as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                for j in range(d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'groupA' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def hyperplane(self, type='slow'):

        n = 100000
        d = 10 + 1

        X = []
        Y = []

        with open('data/synthetic/hyperplane_{0}.arff'.format(type)) as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                for j in range(d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'class1' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == 100000

        return X, Y, n, d

    # real world datasets :
    def powerSupply(self):

        n = 29928
        d = 2 + 1

        drift_times = [17, 47, 76]

        X = []
        Y = []

        with open('data/real-world/powersupply.arff') as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if int(fields[2]) < 12 else -1
                features = {0: 1}
                for j in range(2):
                    features[j+1] = float(fields[j])
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        max_0 = max(X[i][1] for i in range(len(X)))
        max_1 = max(X[i][2] for i in range(len(X)))
        for i in range(len(X)):
            X[i][1] = X[i][1]/max_0
            X[i][2] = X[i][2]/max_1


        return X, Y, n, d, drift_times

    def elec(self):
        #   features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer

        n = 45312
        d = 13 + 1

        drift_times = [20]

        X = []
        Y = []
        with open('data/real-world/electricity-normalized.csv') as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if fields[len(fields)-1] == 'UP' else -1

                features = {0:1}
                features[int(fields[1])] = 1
                for j in range(2, len(fields)-1):
                    features[j+6] = float(fields[j])
                X.append(features)
                Y.append(label)
                i += 1

        assert len(X) == n

        return X, Y, n, d, drift_times

    def airline(self, num2=58100, num1=0):

        # n = 5810462
        n = num2 - num1 # read a partial of this dataset
        d = 679

        drift_times = [31, 67]

        X = []
        Y = []
        with open('data/real-world/airline_2008.data') as file:
            i = 0
            for line in file:
                if  num1 <= i < num2:
                    fields = line.strip().split(',')
                    label = int(fields[len(fields)-1])

                    features = {0:1}
                    for j in range(len(fields)-1):
                        (index, val) = fields[j].split(':')
                        features[int(index)] = float(val)
                    X.append(features)
                    Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    # semi-synthetic datasets :
    def rcv(self, drift_portion=0.3):

        n = 20242
        d = 47237
        drift_times = [30, 60]

        X = []
        Y = []

        with open('data/semi-synthetic/rcv_shuf.data') as file:
            i = 0
            for line in file:
                fields = line.strip().split(' ')
                label = int(fields[0])
                features = {0: 1}
                for j in range(1, len(fields)):
                    (index, val) = fields[j].split(':')
                    features[int(index)] = float(val)
                if n * drift_portion <= i < 2 * n * drift_portion:
                        label *= -1
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def covtype(self, drift_portion=0.3):

        n = 581012
        d = 54 + 1
        drift_times = [30, 60]

        X = []
        Y = []

        with open('data/semi-synthetic/covtype_shuf.data') as file:
            i = 0
            for line in file:
                fields = line.strip().split(' ')
                label = 1 if int(fields[0]) == 1 else -1
                features = {0: 1}
                for j in range(1, len(fields)):
                    (index, val) = fields[j].split(':')
                    features[int(index)] = float(val)
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        # for a drift of rate 0.4
        (i, j, theta) = (1, 8, 3.14)
        Rx = X[:]
        for k in range(len(Rx)):
            if len(Rx) * drift_portion <= k < 2 * len(Rx) * drift_portion:
                x = X[k].copy()
                xi = x[i] if i in x else 0
                xj = x[j] if j in x else 0
                x[i] = xi * math.cos(theta) - xj * math.sin(theta)
                x[j] = xi * math.sin(theta) + xj * math.cos(theta)
                Rx[k] = x
        X = Rx[:]

        return X, Y, n, d, drift_times