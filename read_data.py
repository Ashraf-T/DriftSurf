import numpy
import csv
import os
from collections import defaultdict
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def airline_moa():
    n = 539383
    d = 6

    X = []
    Y = []

    j = 3
    feature_names = {}
    with open('data/airline/airlines.arff') as file:
        i = 1

        for line in file:
            # airline names :
            if i == 3:
                fields = line.strip().split(' ')
                Airline = fields[2][1:-1].split(',')
                for k in Airline:
                    if k+'_air' not in feature_names.keys():
                        feature_names[k+'_air'] = j
                        j += 1
            if i == 5:
                fields = line.strip().split(' ')
                AirportFrom = fields[2][1:-1].split(',')
                for k in AirportFrom:
                    if k+'_from' not in feature_names.keys():
                        feature_names[k[1:-1]+'_from'] = j
                        j += 1
            if i == 6:
                fields = line.strip().split(' ')
                AirportTo = fields[2][1:-1].split(',')
                for k in AirportTo:
                    if k+'_to' not in feature_names.keys():
                        feature_names[k[1:-1]+'_to'] = j
                        j += 1
            if i == 7:
                fields = line.strip().split(' ')
                DayOfWeek = fields[2][1:-1].split(',')
                for k in DayOfWeek:
                    if k+'_day' not in feature_names.keys():
                        feature_names[k+'_day'] = j
                        j += 1
                # print(feature_names)
            if i > 12:
                fields = line.strip().replace(" ",'').split(',')
                label = 1 if int(fields[len(fields) - 1]) ==  1 else -1

                features = {0: 1}
                features[feature_names[fields[0]+'_air']] = 1
                features[feature_names[fields[2]+'_from']] = 1
                features[feature_names[fields[3]+'_to']] = 1
                features[feature_names[fields[4]+'_day']] = 1
                features[1] = float(fields[5])
                features[2] = float(fields[6])

                X.append(features)
                Y.append(label)
            i += 1
    assert len(X) == n

    d = j
    max_1 = max(X[i][1] for i in range(len(X)))
    max_2 = max(X[i][2] for i in range(len(X)))
    for i in range(len(X)):
        X[i][1] = X[i][1]/max_1
        X[i][2] = X[i][2]/max_2

    # n_train = int(0.9 * n)
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    return X, Y, n, d

def sea():
    n = 100000
    d = 4

    X = []
    Y = []

    with open('data/Tornado/SEA.arff') as file:
        i = 0

        for line in file:
            if i > 6:
                fields = line.strip().split(',')
                label = 1 if (fields[len(fields) - 1]) == 'p' else -1
                features = {0: 1}
                for j in range(len(fields) - 1):
                    features[j + 1] = float(fields[j])

                X.append(features)
                Y.append(label)
            i += 1
    assert len(X) == n

    max_0 = max(X[i][1] for i in range(len(X)))
    max_1 = max(X[i][2] for i in range(len(X)))
    max_2 = max(X[i][3] for i in range(len(X)))
    for i in range(len(X)):
        X[i][1] = X[i][1]/max_0
        X[i][2] = X[i][2]/max_1
        X[i][3] = X[i][3]/max_2

    # n_train = int(0.9 * n)
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    return X, Y, n, d

def circles_new():
    # n = 10000
    n = 20000
    d = 3

    X = []
    Y = []

    # with open('data/Circles.arff') as file:
    with open('data/Circles_20000.arff') as file:
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

    # n_train = int(0.9 * n)
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    return X, Y, n, d

def circles():
    n = 100000
    d = 3

    X = []
    Y = []

    with open('data/Tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_101.arff') as file:
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

    # n_train = int(0.9 * n)
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    return X, Y, n, d

def mixed():

    n = 100000
    d = 5

    X = []
    Y = []

    with open('data/Tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_101.arff') as file:
        i = 0

        for line in file:
            if i > 7:
                fields = line.strip().split(',')
                label = 1 if (fields[len(fields) - 1]) == 'p' else -1
                features = {0: 1}
                features[1] = 1 if (fields[0] == 'True') else 0
                features[2] = 1 if (fields[1] == 'True') else 0
                for j in range(2,len(fields) - 1):
                    features[j + 1] = float(fields[j])

                X.append(features)
                Y.append(label)
            i += 1
    assert len(X) == n

        # n_train = int(0.9 * n)
        # train_data = data[:n_train]
        # test_data = data[n_train:]

    return X, Y, n, d

def sine1_new():

    n = 10000
    d = 3

    X = []
    Y = []

    with open('data/sine1.arff') as file:
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

        # n_train = int(0.9 * n)
        # train_data = data[:n_train]
        # test_data = data[n_train:]

    return X, Y, n, d

def sine1():

    n = 100000
    d = 3

    X = []
    Y = []

    with open('data/Tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.arff') as file:
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

        # n_train = int(0.9 * n)
        # train_data = data[:n_train]
        # test_data = data[n_train:]

    return X, Y, n, d

def Luxembourgcal():
    n = 1901
    d = 32

    X = []
    Y = []

    with open('data/Luxembourg/LUweka.csv') as file:
        i = 0
        for line in file:
            if i > 0:
                fields = line.strip().split(',')
                label = 1 if int(fields[len(fields)-1]) == 1 else -1
                features = {0: 1}
                for j in range(len(fields)-1):
                    features[j+1] = float(fields[j])

                X.append(features)
                Y.append(label)
            i += 1
    assert len(X) == n

    # n_train = int(0.9 * n)
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    # print(X[2])
    # print(Y[2])
    return X, Y, n, d

def powerSupply():
    n = 29928
    d = 3

    X = []
    Y = []

    with open('data/PowerSupply/powersupply.arff.txt') as file:
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
            # print(i)
    assert len(X) == n

    max_0 = max(X[i][1] for i in range(len(X)))
    max_1 = max(X[i][2] for i in range(len(X)))
    for i in range(len(X)):
        X[i][1] = X[i][1]/max_0
        X[i][2] = X[i][2]/max_1

    # print(X[0])

    # n_train = int(0.9 * n)
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    return X, Y, n, d

def airline_preprocess1():
    # attributes:
    # 1. Year : 1988 - 2008 (cat)
    # 2. Month : 1 - 12 (cat)
    # 3. Day of Month 1 - 30 (cat)
    # 4. Day of Week : 1 - 7 (cat)
    # 5. CRS Departure Time: (float)
    # 6. CRS Arrival Time: (float)
    # 7. Unique Carrier: (cat)
    # 8. Flight Number: remove
    # 9. Actual Elapse Time: (float)
    # 10.Origin: (cat)
    # 11.Destination (cat)
    # 12.Distance (float)
    # 13.Diverted (bool)

    # target variable: Arrival Delay, given in seconds -> > 0 : delayed else not

    column_names = ['year', 'month', 'day_of_month', 'day_of_week',
                    'CRS_departure_time', 'CRS_arrival_time', 'carrier',
                    'flight_num', 'elapsed_time', 'origin', 'dest',
                    'distance', 'diverted', 'delay']

    # pre-processing
    dataset = pd.read_csv('data/airline/2008_14col.data', sep = ',', names=column_names, index_col=False)

    min_max_scaler = MinMaxScaler()
    column_names_to_normalize = ['CRS_departure_time', 'CRS_arrival_time', 'elapsed_time', 'distance']
    x = dataset[column_names_to_normalize].values
    x_scaled = min_max_scaler.fit_transform(x)
    dataset_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index=dataset.index)
    dataset[column_names_to_normalize] = dataset_temp

    print(dataset.head())

    column_names_to_categorize = ['year', 'month', 'day_of_month', 'day_of_week', 'carrier', 'origin', 'dest']
    for col in column_names_to_categorize:
        dataset = pd.concat([dataset, pd.get_dummies(dataset[col], prefix=col)], axis=1)
        dataset.drop([col], axis=1, inplace=True)

    print(dataset.head())

    with open('data/airline_col.txt', 'w') as f:
        f.write(",".join(list(dataset.columns)))


def airline_preprocess2():

    X = []
    Y = []
    column_names = ['year', 'month', 'day_of_month', 'day_of_week', 'CRS_departure_time', 'CRS_arrival_time', 'carrier',
                    'flight_num', 'elapsed_time', 'origin', 'dest', 'distance', 'diverted', 'delay']

    column_names_to_normalize = ['CRS_departure_time', 'CRS_arrival_time', 'elapsed_time', 'distance']
    column_names_to_categorize = ['year', 'month', 'day_of_month', 'day_of_week', 'carrier', 'origin', 'dest']



    with open('data/airline_col.txt') as f:
        for line in f:
            columns = line.split(',')

    features_dict = {}
    c = 1
    for col in columns:
        features_dict[col] = c
        c += 1

    # print(features_dict)
    n = 5810462
    d = len(features_dict.keys()) + 1
    data = []
    with open('data/airline/2008_14col.data') as file:
        i = 0
        for line in file:

            fields = line.strip().split(',')
            label = 1 if float(fields[len(fields)-1]) > 0 else -1
            features = {}
            for j in range(len(column_names)-1):
                if column_names[j] in column_names_to_categorize:
                    features[features_dict[column_names[j]+'_'+fields[j]]] = 1
                elif column_names[j] != 'flight_num':
                    features[features_dict[column_names[j]]] = float(fields[j])

            # data.append((i, features, label))
            X.append(features)
            Y.append(label)
            i += 1
    #         # print(i)
    assert i == n

    for col in column_names_to_normalize:
        max_col = max(X[i][features_dict[col]] for i in range(len(X)))

        for i in range(len(X)):
            X[i][features_dict[col]] = X[i][features_dict[col]]/max_col


    data = [str(features_i)[1:-1] + ', ' + str(label_i) for features_i, label_i in zip(X, Y)]
    with open('data/airline/2008_14col_processed.data', 'w') as file:
        file.write('\n'.join(data))


def airline():

    X = []
    Y = []

    n = 5810462
    d = 679

    print(n)
    # data = []
    with open('data/airline/2008_14col_processed.data') as file:
        i = 0
        for line in file:
            fields = line.strip().split(',')
            label = int(fields[len(fields)-1])

            features = {0:1}
            for j in range(len(fields)-1):
                (index, val) = fields[j].split(':')
                features[int(index)] = float(val)
            # data.append((i, features, label))
            X.append(features)
            Y.append(label)
            i += 1

    assert len(X) == n


    # print(X[0], Y[0])
    print(n, d)

    return X, Y, n, d


def airline_trim(num2, num1=0):

    X = []
    Y = []

    # n = 5810462
    n = num2 - num1
    d = 679

    print(n)
    # data = []
    with open('data/airline/2008_14col_processed.data') as file:
        i = 0
        for line in file:
            if  num1 <= i < num2:
                fields = line.strip().split(',')
                label = int(fields[len(fields)-1])

                features = {0:1}
                for j in range(len(fields)-1):
                    (index, val) = fields[j].split(':')
                    features[int(index)] = float(val)
                # data.append((i, features, label))
                X.append(features)
                Y.append(label)
            i += 1
    # print(len(X), n)
    assert len(X) == n


    # print(X[0], Y[0])
    print(n, d)

    return X, Y, n, d


def elec():
    #   features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer

    n = 45312
    d = 14

    X = []
    Y = []
    with open('data/elec/electricity-normalized.csv') as file:
        i = 0
        for line in file:
            fields = line.strip().split(',')
            label = 1 if fields[len(fields)-1] == 'UP' else -1

            features = {0:1}
            features[int(fields[1])] = 1
            for j in range(2, len(fields)-1):
                features[j+6] = float(fields[j])
            # data.append((i, features, label))
            X.append(features)
            Y.append(label)
            i += 1

    assert len(X) == n

    return X, Y, n, d


if __name__ == "__main__":
    # Luxembourgcal()
    # powerSupply()
    # airline()
    # airline_trim(1700*581, 1900*581)
    # elec()
    # sine1()
    # mixed()
    # airline_moa()
    sine1_new()