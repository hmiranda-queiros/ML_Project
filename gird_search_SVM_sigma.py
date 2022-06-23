# --- modules --- #
import time
import matplotlib.pyplot as plt
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
import csv
from itertools import zip_longest

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
# nb_raw = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
# nb_lle = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
# nb_mlle = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
nb_raw = np.logspace(-5, -3, 20, endpoint=True)
nb_lle = np.logspace(3, 5, 20, endpoint=True)
nb_mlle = np.logspace(2, 4, 20, endpoint=True)
# # raw
elapsed_tot_raw = []
f1_scores_raw = []
accuracy_scores_raw = []
cfm_raw = []
states_raw = []
for i in nb_raw:
    classifier = SVC(C=1.0, kernel='rbf', gamma=i, coef0=0)
    start_time = time.time()
    classifier.fit(X_train_dict["RAW"], y_train)
    predictions = classifier.predict(X_test_dict["RAW"])
    elapsed_time = time.time() - start_time
    elapsed_tot_raw.append(elapsed_time + elapsed_dict['RAW'])
    states_raw.append("RAW:Sigma:" + str(i) + ';t:' + f'{elapsed_tot_raw[-1]:.2f}')
    f1_scores_raw.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_raw.append(metrics.accuracy_score(y_test, predictions))
    cfm_raw.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print("RAW:Sigma:" + str(i) + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # lle
elapsed_clf_lle = []
elapsed_tot_lle = []
f1_scores_lle = []
accuracy_scores_lle = []
cfm_lle = []
states_lle = []
for i in nb_lle:
    classifier = SVC(C=1.0, kernel='rbf', gamma=i, coef0=0)
    start_time = time.time()
    classifier.fit(X_train_dict["LLE"], y_train)
    predictions = classifier.predict(X_test_dict["LLE"])
    elapsed_time = time.time() - start_time
    elapsed_clf_lle.append(elapsed_time)
    elapsed_tot_lle.append(elapsed_time + elapsed_dict['LLE'])
    states_lle.append("LLE:Sigma:" + str(i) + ';t:' + f'{elapsed_tot_lle[-1]:.2f}')
    f1_scores_lle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_lle.append(metrics.accuracy_score(y_test, predictions))
    cfm_lle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print("LLE:Sigma:" + str(i) + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # mlle
elapsed_clf_mlle = []
elapsed_tot_mlle = []
f1_scores_mlle = []
accuracy_scores_mlle = []
cfm_mlle = []
states_mlle = []
for i in nb_mlle:
    classifier = SVC(C=1.0, kernel='rbf', gamma=i, coef0=0)
    start_time = time.time()
    classifier.fit(X_train_dict["MLLE"], y_train)
    predictions = classifier.predict(X_test_dict["MLLE"])
    elapsed_time = time.time() - start_time
    elapsed_clf_mlle.append(elapsed_time)
    elapsed_tot_mlle.append(elapsed_time + elapsed_dict['MLLE'])
    states_mlle.append("MLLE:Sigma:" + str(i) + ';t:' + f'{elapsed_tot_mlle[-1]:.2f}')
    f1_scores_mlle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_mlle.append(metrics.accuracy_score(y_test, predictions))
    cfm_mlle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print("MLLE:Sigma:" + str(i) + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')

# --- plots --- #
x_lle = list(range(len(states_lle)))
y_lle = f1_scores_lle
z_lle = accuracy_scores_lle
x_mlle = list(range(len(states_mlle)))
y_mlle = f1_scores_mlle
z_mlle = accuracy_scores_mlle
x_raw = list(range(len(states_raw)))
y_raw = f1_scores_raw
z_raw = accuracy_scores_raw
fig, ax = plt.subplots(2, 1, figsize=(60, 30))
ax[0].plot(x_lle, y_lle, 'o-')
ax[0].plot(x_raw, y_raw, 'x-', color='r')
ax[0].set_xticks(x_lle)
ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
ax[0].grid()
ax[0].legend(['lle', 'raw'], fontsize=20)
ax[0].set_title('F1-score LLE', fontsize=30)
ax[0].tick_params(axis='y', which='major', labelsize=20)
ax[1].plot(x_mlle, y_mlle, 'x-')
ax[1].plot(x_raw, y_raw, 'o-', color='r')
ax[1].set_xticks(x_mlle)
ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
ax[1].grid()
ax[1].legend(['mlle', 'raw'], fontsize=20)
ax[1].set_title('F1-score MLLE', fontsize=30)
ax[1].tick_params(axis='y', which='major', labelsize=20)
fig.savefig(f'./plots/f1_{nb_sample * 2}_samples.png')
fig, ax = plt.subplots(2, 1, figsize=(60, 30))
ax[0].plot(x_lle, z_lle, 'o-')
ax[0].plot(x_raw, z_raw, 'x-', color='r')
ax[0].set_xticks(x_lle)
ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
ax[0].grid()
ax[0].legend(['lle', 'raw'], fontsize=20)
ax[0].set_title('Accur LLE', fontsize=30)
ax[0].tick_params(axis='y', which='major', labelsize=20)
ax[1].plot(x_mlle, z_mlle, 'x-')
ax[1].plot(x_raw, z_raw, 'o-', color='r')
ax[1].set_xticks(x_mlle)
ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
ax[1].grid()
ax[1].legend(['mlle', 'raw'], fontsize=20)
ax[1].set_title('Accur MLLE', fontsize=30)
ax[1].tick_params(axis='y', which='major', labelsize=20)
fig.savefig(f'./plots/accur_{nb_sample * 2}_samples.png')
fig, ax = plt.subplots(3, max(len(nb_lle), len(nb_mlle), len(nb_raw)), figsize=(120, 30))
for i in range(len(nb_raw)):
    sns.heatmap(cfm_raw[i], annot=True, ax=ax[0, i])
    ax[0, i].set_title(states_raw[i], fontsize=20)
for i in range(len(nb_lle)):
    sns.heatmap(cfm_lle[i], annot=True, ax=ax[1, i])
    ax[1, i].set_title(states_lle[i], fontsize=20)
for i in range(len(nb_mlle)):
    sns.heatmap(cfm_mlle[i], annot=True, ax=ax[2, i])
    ax[2, i].set_title(states_mlle[i], fontsize=20)
fig.savefig(f'./plots/cfm_{nb_sample * 2}_samples.png.png')

# --- csv files --- #
nb_sigma_lle = nb_lle.tolist()
nb_sigma_mlle = nb_mlle.tolist()
nb_sigma_raw = nb_raw.tolist()
data = [nb_sigma_lle, f1_scores_lle, accuracy_scores_lle, elapsed_clf_lle, elapsed_tot_lle,
        nb_sigma_mlle, f1_scores_mlle, accuracy_scores_mlle, elapsed_clf_mlle, elapsed_tot_mlle,
        nb_sigma_raw, f1_scores_raw, accuracy_scores_raw, elapsed_tot_raw]
export_data = zip_longest(*data, fillvalue='')
with open(f'./csv/grd_sh_SVM_sigma_{nb_sample * 2}.csv', 'w', encoding="ISO-8859-1", newline='') as file:
    write = csv.writer(file)
    write.writerow(("nb_sigma_lle", "f1_scores_lle", "accuracy_scores_lle", "elapsed_clf_lle", "elapsed_tot_lle",
                    "nb_sigma_mlle", "f1_scores_mlle", "accuracy_scores_mlle", "elapsed_clf_mlle", "elapsed_tot_mlle",
                    "nb_sigma_raw", "f1_scores_raw", "accuracy_scores_raw", "elapsed_tot_raw"))
    write.writerows(export_data)
