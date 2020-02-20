Summary of files:
- /data directory includes 3 folders for synthetic, semi-syntetic, and real-world datasets that we used
- read_data.py contains the class for reading datasets
- models.py contains the methods for SGD, STRSAGA, DriftSurf, and AUE on Logistic Regression model
- training.py contains the training process of DriftSurf, MDDM, AUE, and Aware, using two different base learners SGD and STRSAGA.
- hyperparameters.py contains the hyper-parameters used for training over the used datasets. 
- main.py runs the experiment necessary to produce the results shown in Table 2 and also time series plots shown in Figure 1,2, and 3. It takes four arguments, the dataset, the way we want to limit computational power (for each learner or for each algorithm), processing rate rho/m, and base learner. For these input parameters, it outputs time series plot and also prints out the average of misclassification rate over time for each algorithm.
- sensitivity_noise.py runs the experiment necessary to produce the results shown in Figure 4. 
- greedy.py runs the experiment necessary to produce the results shown in Table 11 and Figure 11 in appendix. It takes one argument: name of dataset, it outputs time series plot and also prints out the average of misclassification rate over time for DriftSurf with greedy and without greedy approaches.
- results.py contains the methods to produce plots and results of these experiments.