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
nb_lle = np.arange(8, 19, 1)
nb_mlle = np.arange(18, 29, 1)
methods = OrderedDict()
elapsed_dr_lle = OrderedDict()
elapsed_dr_mlle = OrderedDict()
X_train_dict_lle = OrderedDict()
X_test_dict_lle = OrderedDict()
X_train_dict_mlle = OrderedDict()
X_test_dict_mlle = OrderedDict()
labels_dr_lle = []
labels_dr_mlle = []
for i in range(len(nb_lle)):
    labels_dr_lle.append('LLE:cmp:' + str(nb_lle[i]) + ";ngh:" + str(nb_lle[i] + 2))
for i in range(len(nb_mlle)):
    labels_dr_mlle.append('MLLE:cmp:' + str(nb_mlle[i]) + ";ngh:" + str(nb_mlle[i] + 2))
LLE = partial(manifold.LocallyLinearEmbedding,
              eigen_solver='auto',
              neighbors_algorithm='auto',
              random_state=617)
for i in range(len(nb_lle)):
    methods[labels_dr_lle[i]] = LLE(n_components=nb_lle[i], n_neighbors=nb_lle[i] + 2, method="standard")
    start_time = time.time()
    X_train_dict_lle[labels_dr_lle[i]] = methods[labels_dr_lle[i]].fit_transform(X_train)
    X_test_dict_lle[labels_dr_lle[i]] = methods[labels_dr_lle[i]].transform(X_test)
    elapsed_time = time.time() - start_time
    elapsed_dr_lle[labels_dr_lle[i]] = elapsed_time
    print(labels_dr_lle[i] + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
for i in range(len(nb_mlle)):
    methods[labels_dr_mlle[i]] = LLE(n_components=nb_mlle[i], n_neighbors=nb_mlle[i] + 2, method="modified")
    start_time = time.time()
    X_train_dict_mlle[labels_dr_mlle[i]] = methods[labels_dr_mlle[i]].fit_transform(X_train)
    X_test_dict_mlle[labels_dr_mlle[i]] = methods[labels_dr_mlle[i]].transform(X_test)
    elapsed_time = time.time() - start_time
    elapsed_dr_mlle[labels_dr_mlle[i]] = elapsed_time
    print(labels_dr_mlle[i] + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')

# --- classification --- #
classifier = SVC(C=1.0, kernel='rbf', gamma='scale', coef0=0)
# # raw
start_time = time.time()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
elapsed_time_raw = time.time() - start_time
states_raw = 'RAW:t:' + f'{elapsed_time_raw:.2f}'
f1_scores_raw = metrics.f1_score(y_test, predictions)
accuracy_scores_raw = metrics.accuracy_score(y_test, predictions)
cfm_raw = metrics.confusion_matrix(y_test, predictions, normalize='true')
print('RAW finished in ' + f'{elapsed_time_raw:.2f}' + ' s!')
# # lle
elapsed_clf_lle = []
elapsed_tot_lle = []
f1_scores_lle = []
accuracy_scores_lle = []
cfm_lle = []
states_lle = []
for label in labels_dr_lle:
    start_time = time.time()
    classifier.fit(X_train_dict_lle[label], y_train)
    predictions = classifier.predict(X_test_dict_lle[label])
    elapsed_time = time.time() - start_time
    elapsed_clf_lle.append(elapsed_time)
    elapsed_tot_lle.append(elapsed_time + elapsed_dr_lle[label])
    states_lle.append(label + ';t:' + f'{elapsed_tot_lle[-1]:.2f}')
    f1_scores_lle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_lle.append(metrics.accuracy_score(y_test, predictions))
    cfm_lle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # mlle
elapsed_clf_mlle = []
elapsed_tot_mlle = []
f1_scores_mlle = []
accuracy_scores_mlle = []
cfm_mlle = []
states_mlle = []
for label in labels_dr_mlle:
    start_time = time.time()
    classifier.fit(X_train_dict_mlle[label], y_train)
    predictions = classifier.predict(X_test_dict_mlle[label])
    elapsed_time = time.time() - start_time
    elapsed_clf_mlle.append(elapsed_time)
    elapsed_tot_mlle.append(elapsed_time + elapsed_dr_mlle[label])
    states_mlle.append(label + ';t:' + f'{elapsed_tot_mlle[-1]:.2f}')
    f1_scores_mlle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_mlle.append(metrics.accuracy_score(y_test, predictions))
    cfm_mlle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')

# --- plots --- #
x_lle = list(range(len(labels_dr_lle)))
y_lle = f1_scores_lle
z_lle = accuracy_scores_lle
x_mlle = list(range(len(labels_dr_mlle)))
y_mlle = f1_scores_mlle
z_mlle = accuracy_scores_mlle
fig, ax = plt.subplots(2, 1, figsize=(60, 30))
ax[0].plot(x_lle, y_lle, 'o-')
ax[0].axhline(y=f1_scores_raw, color='r', linestyle='-', linewidth=2)
ax[0].set_xticks(x_lle)
ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
ax[0].grid()
ax[0].legend(['lle', 'raw'], fontsize=20)
ax[0].set_title('F1-score LLE', fontsize=30)
ax[0].tick_params(axis='y', which='major', labelsize=20)
ax[1].plot(x_mlle, y_mlle, 'x-')
ax[1].axhline(y=f1_scores_raw, color='r', linestyle='-', linewidth=2)
ax[1].set_xticks(x_mlle)
ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
ax[1].grid()
ax[1].legend(['mlle', 'raw'], fontsize=20)
ax[1].set_title('F1-score MLLE', fontsize=30)
ax[1].tick_params(axis='y', which='major', labelsize=20)
fig.savefig(f'./plots/f1_{nb_sample * 2}_samples.png')
fig, ax = plt.subplots(2, 1, figsize=(60, 30))
ax[0].plot(x_lle, z_lle, 'o-')
ax[0].axhline(y=accuracy_scores_raw, color='r', linestyle='-', linewidth=2)
ax[0].set_xticks(x_lle)
ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
ax[0].grid()
ax[0].legend(['lle', 'raw'], fontsize=20)
ax[0].set_title('Accur LLE', fontsize=30)
ax[0].tick_params(axis='y', which='major', labelsize=20)
ax[1].plot(x_mlle, z_mlle, 'x-')
ax[1].axhline(y=accuracy_scores_raw, color='r', linestyle='-', linewidth=2)
ax[1].set_xticks(x_mlle)
ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
ax[1].grid()
ax[1].legend(['mlle', 'raw'], fontsize=20)
ax[1].set_title('Accur MLLE', fontsize=30)
ax[1].tick_params(axis='y', which='major', labelsize=20)
fig.savefig(f'./plots/accur_{nb_sample * 2}_samples.png')
fig, ax = plt.subplots(3, max(len(nb_lle), len(nb_mlle)), figsize=(120, 30))
sns.heatmap(cfm_raw, annot=True, ax=ax[0, 0])
ax[0, 0].set_title(states_raw, fontsize=20)
for i in range(len(nb_lle)):
    sns.heatmap(cfm_lle[i], annot=True, ax=ax[1, i])
    ax[1, i].set_title(states_lle[i], fontsize=20)
for i in range(len(nb_mlle)):
    sns.heatmap(cfm_mlle[i], annot=True, ax=ax[2, i])
    ax[2, i].set_title(states_mlle[i], fontsize=20)
fig.savefig(f'./plots/cfm_{nb_sample * 2}_samples.png')

# --- csv files --- #
nb_compo_lle = nb_lle.tolist()
nb_compo_mlle = nb_mlle.tolist()
f1_scores_raw = [f1_scores_raw for i in range(max(len(nb_compo_lle), len(nb_compo_mlle)))]
accuracy_scores_raw = [accuracy_scores_raw for i in range(max(len(nb_compo_lle), len(nb_compo_mlle)))]
elapsed_times_raw = [elapsed_time_raw for i in range(max(len(nb_compo_lle), len(nb_compo_mlle)))]
data = [nb_compo_lle, f1_scores_lle, accuracy_scores_lle, list(elapsed_dr_lle.values()), elapsed_clf_lle, elapsed_tot_lle,
        nb_compo_mlle, f1_scores_mlle, accuracy_scores_mlle, list(elapsed_dr_mlle.values()), elapsed_clf_mlle, elapsed_tot_mlle,
        f1_scores_raw, accuracy_scores_raw, elapsed_times_raw]
export_data = zip_longest(*data, fillvalue='')
with open(f'./csv/grd_sh_components_{nb_sample * 2}.csv', 'w', encoding="ISO-8859-1", newline='') as file:
    write = csv.writer(file)
    write.writerow(("nb_compo_lle", "f1_scores_lle", "accuracy_scores_lle", "elapsed_dr_lle", "elapsed_clf_lle", "elapsed_tot_lle",
                    "nb_compo_mlle", "f1_scores_mlle", "accuracy_scores_mlle", "elapsed_dr_mlle", "elapsed_clf_mlle", "elapsed_tot_mlle",
                    "f1_scores_raw", "accuracy_scores_raw", "elapsed_times_raw"))
    write.writerows(export_data)
