# Imports
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
# Setting up a data stream
stream = SEAGenerator(random_state=1)
# Setup Hoeffding Tree estimator
ht = HoeffdingTreeClassifier()
# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 200
# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = ht.predict(X)
    if y[0] == y_pred[0]:
        correct_cnt += 1
    ht = ht.partial_fit(X, y)
    n_samples += 1
# Display results
print(ht.predict_proba(X)[0][1])
print(ht.predict(X))
print(X.shape)
print(y.shape)
print('{} samples analyzed.'.format(n_samples))
print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples))