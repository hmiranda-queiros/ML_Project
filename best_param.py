# --- modules --- #
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict
from functools import partial
import pandas as pd
import seaborn as sns  # sns.set()
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.mixture import GaussianMixture


# --- preprocessing --- #
data = pd.read_csv("./data/data.csv")

# # remove empty feature
data.drop(['Unnamed: 83'], axis=1, inplace=True)

# # remove useless features
data.drop(['encounter_id', 'patient_id', 'hospital_id',
           'icu_admit_source', 'icu_id', 'icu_stay_type', 'icu_type',
           'pre_icu_los_days', 'apache_post_operative', 'gcs_unable_apache',
           'apache_3j_bodysystem', 'apache_2_bodysystem'
           ], axis=1, inplace=True)

# # remove rows with at least one null value in all features
data = data[data.isna().sum(axis=1) == 0]

# # Converting categorical values to numerical ones
for item in ['ethnicity', 'gender']:
    data[item] = data[item].astype('category')
    data[item + '_num'] = data[item].cat.codes
    data.drop(item, axis=1, inplace=True)
data.to_csv('./data/data_pp.csv', index=False)
print(f'Original: #data = {data.shape[0]}, #features = {data.shape[1]}')
nb_patients_org = len(data['hospital_death'])
nb_survived_org = len(data[data['hospital_death'] == 0])
nb_died_org = len(data[data['hospital_death'] == 1])
death_proportion_org = nb_died_org / nb_patients_org
print(
    f'Original: #patients = {nb_patients_org}, #survived = {nb_survived_org}, #died = {nb_died_org}, proportion = {death_proportion_org:.3f}')

# --- sampling --- #
data_died = data[data['hospital_death'] == 1]
data_survived = data[data['hospital_death'] == 0]
nb_sample = nb_died_org
data_died = data_died.sample(nb_sample, random_state=617)
data_survived = data_survived.sample(nb_sample, random_state=617)
data_sp = pd.concat([data_died, data_survived], ignore_index=True)
# data_sp.to_csv('./data/data_sp.csv', index=False)
print(f'Sampled: #data = {data_sp.shape[0]}, #features = {data_sp.shape[1]}')
nb_patients_sp = len(data_sp['hospital_death'])
nb_survived_sp = len(data_sp[data_sp['hospital_death'] == 0])
nb_died_sp = len(data_sp[data_sp['hospital_death'] == 1])
print(f'Sampled: #patients = {nb_patients_sp}, #survived = {nb_survived_sp}, #died = {nb_died_sp}')

# # splitting to train and test
y = data_sp['hospital_death']
X = data_sp.drop(['hospital_death'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, stratify=y, random_state=617)

# # normalizing train and test
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(f'Train Split: #data = {X_train.shape[0]}, #features = {X_train.shape[1]}')
nb_patients_spl = len(y_train)
nb_survived_spl = len(y_train[y_train == 0])
nb_died_spl = len(y_train[y_train == 1])
death_proportion_spl = nb_died_spl / nb_patients_spl
print(
    f'Train Split: #patients = {nb_patients_spl}, #survived = {nb_survived_spl}, #died = {nb_died_spl}, proportion = {death_proportion_spl:.3f}')

# --- dimensionality reduction --- #
methods = OrderedDict()
methods['RAW'] = []
X_train_dict = OrderedDict()
X_train_dict['RAW'] = X_train
X_test_dict = OrderedDict()
X_test_dict['RAW'] = X_test
elapsed_dict = OrderedDict()
elapsed_dict['RAW'] = 0
LLE = partial(manifold.LocallyLinearEmbedding,
              eigen_solver='auto',
              neighbors_algorithm='auto',
              random_state=617)
methods['LLE'] = LLE(n_components=12, n_neighbors=14, method="standard")
start_time = time.time()
X_train_dict['LLE'] = methods['LLE'].fit_transform(X_train)
X_test_dict['LLE'] = methods['LLE'].transform(X_test)
elapsed_time = time.time() - start_time
elapsed_dict['LLE'] = elapsed_time
print('LLE' + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
methods['MLLE'] = LLE(n_components=19, n_neighbors=22, method="modified")
start_time = time.time()
X_train_dict['MLLE'] = methods['MLLE'].fit_transform(X_train)
X_test_dict['MLLE'] = methods['MLLE'].transform(X_test)
elapsed_time = time.time() - start_time
elapsed_dict['MLLE'] = elapsed_time
print('MLLE' + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')

# --- classification --- #
states = []
elapsed = []
elapsed_clf = []
f1_scores = []
accuracy_scores = []
cfms = []
for method in methods:
    if method == "RAW":
        classifier = SVC(C=1.0, kernel='rbf', gamma=0.00009, coef0=0)
    elif method == "LLE":
        classifier = SVC(C=1.0, kernel='rbf', gamma=8859, coef0=0)
    else:
        classifier = SVC(C=1.0, kernel='rbf', gamma=695, coef0=0)
    start_time = time.time()
    classifier.fit(X_train_dict[method], y_train)
    predictions = classifier.predict(X_test_dict[method])
    elapsed_time = time.time() - start_time
    elapsed_clf.append(elapsed_time)
    states.append(
        method + ':SVM:' + f'{elapsed_time:.2f}' + ':' + f'{elapsed_time + elapsed_dict[method]:.2f}')
    print(states[-1])
    elapsed.append(elapsed_time + elapsed_dict[method])
    f1_scores.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores.append(metrics.accuracy_score(y_test, predictions))
    cfms.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
result = {'states': states, 'f1_scores': f1_scores, 'accuracy_scores': accuracy_scores, 'cfms': cfms, 'time': elapsed}
result_df = pd.DataFrame(data=result)
result_df.to_csv('./data/results.csv')
ax = result_df.plot(x="states", y=["f1_scores", "accuracy_scores"], kind='bar', figsize=(30, 20))
plt.savefig('./plots/accur_f1.png')

# --- confusion matrices ---#
# raw
fig, ax = plt.subplots()
sns.heatmap(cfms[0], annot=True, ax=ax, annot_kws={"fontsize": 20})
ax.figure.axes[-1].yaxis.label.set_size(30)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
fig.savefig('./plots/cfm_raw.svg')
# lle
fig, ax = plt.subplots()
sns.heatmap(cfms[1], annot=True, ax=ax, annot_kws={"fontsize": 20})
ax.figure.axes[-1].yaxis.label.set_size(30)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
fig.savefig('./plots/cfm_lle.svg')
# mlle
fig, ax = plt.subplots()
sns.heatmap(cfms[2], annot=True, ax=ax, annot_kws={"fontsize": 20})
ax.figure.axes[-1].yaxis.label.set_size(30)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
fig.savefig('./plots/cfm_mlle.svg')
print("states")
print(states)
print("elapsed time for dimensionality reduction")
print(elapsed_dict)
print("elapsed time for classification")
print(elapsed_clf)
print("elapsed time (total)")
print(elapsed)
print("accuracy scores")
print(accuracy_scores)
print("f1 scores")
print(f1_scores)
