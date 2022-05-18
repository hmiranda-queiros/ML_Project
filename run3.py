# --- modules --- #
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial
import pandas as pd
import seaborn as sns       # sns.set()
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
    data[item+'_num'] = data[item].cat.codes
    data.drop(item, axis=1, inplace=True)
data.to_csv('./data/data_pp.csv', index=False)
print(f'Original: #data = {data.shape[0]}, #features = {data.shape[1]}')
nb_patients_org = len(data['hospital_death'])
nb_survived_org = len(data[data['hospital_death'] == 0])
nb_died_org = len(data[data['hospital_death'] == 1])
death_proportion_org = nb_died_org / nb_patients_org
print(f'Original: #patients = {nb_patients_org}, #survived = {nb_survived_org}, #died = {nb_died_org}, proportion = {death_proportion_org:.3f}')


# --- sampling --- #
data_died = data[data['hospital_death'] == 1]
data_survived = data[data['hospital_death'] == 0]
data_survived = data_survived.sample(nb_died_org, random_state=617)
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
print(f'Train Split: #patients = {nb_patients_spl}, #survived = {nb_survived_spl}, #died = {nb_died_spl}, proportion = {death_proportion_spl:.3f}')


# --- dimensionality reduction --- #
nb = np.arange(7, 16, 1)
# nb = np.arange(15, 25, 1)
methods = OrderedDict()
elapsed_dr_lle = OrderedDict()
elapsed_dr_mlle = OrderedDict()
X_train_dict_lle = OrderedDict()
X_test_dict_lle = OrderedDict()
X_train_dict_mlle = OrderedDict()
X_test_dict_mlle = OrderedDict()
labels_dr_lle = []
labels_dr_mlle = []
for i in range(len(nb)):
    labels_dr_lle.append('LLE:' + str(nb[i]))
    labels_dr_mlle.append('MLLE:' + str(nb[i]))
LLE = partial(manifold.LocallyLinearEmbedding,
              eigen_solver='dense',
              neighbors_algorithm='auto',
              n_neighbors=18,
              random_state=617)
for i in range(len(nb)):
    methods[labels_dr_lle[i]] = LLE(n_components=nb[i], method="standard")
    start_time = time.time()
    X_train_dict_lle[labels_dr_lle[i]] = methods[labels_dr_lle[i]].fit_transform(X_train)
    X_test_dict_lle[labels_dr_lle[i]] = methods[labels_dr_lle[i]].transform(X_test)
    elapsed_time = time.time() - start_time
    elapsed_dr_lle[labels_dr_lle[i]] = elapsed_time
    print(labels_dr_lle[i] + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
for i in range(len(nb)):
    methods[labels_dr_mlle[i]] = LLE(n_components=nb[i], method="modified")
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
states_raw = 'RAW:' + f'{elapsed_time_raw:.2f}'
f1_scores_raw = metrics.f1_score(y_test, predictions)
accuracy_scores_raw = metrics.accuracy_score(y_test, predictions)
cfm_raw = metrics.confusion_matrix(y_test, predictions, normalize='true')
print('RAW finished in ' + f'{elapsed_time_raw:.2f}' + ' s!')
# # lle
elapsed_tot_lle = []; f1_scores_lle = []; accuracy_scores_lle = []; cfm_lle = []; states_lle = []
for label in labels_dr_lle:
    start_time = time.time()
    classifier.fit(X_train_dict_lle[label], y_train)
    predictions = classifier.predict(X_test_dict_lle[label])
    elapsed_time = time.time() - start_time
    elapsed_tot_lle.append(elapsed_time+elapsed_dr_lle[label])
    states_lle.append(label + ':' + f'{elapsed_tot_lle[-1]:.2f}')
    f1_scores_lle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_lle.append(metrics.accuracy_score(y_test, predictions))
    cfm_lle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # mlle
elapsed_tot_mlle = []; f1_scores_mlle = []; accuracy_scores_mlle = []; cfm_mlle = []; states_mlle = []
for label in labels_dr_mlle:
    start_time = time.time()
    classifier.fit(X_train_dict_mlle[label], y_train)
    predictions = classifier.predict(X_test_dict_mlle[label])
    elapsed_time = time.time() - start_time
    elapsed_tot_mlle.append(elapsed_time+elapsed_dr_mlle[label])
    states_mlle.append(label + ':' + f'{elapsed_tot_mlle[-1]:.2f}')
    f1_scores_mlle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_mlle.append(metrics.accuracy_score(y_test, predictions))
    cfm_mlle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # plots
x_lle = list(range(len(labels_dr_lle)))
y_lle = f1_scores_lle
z_lle = accuracy_scores_lle
x_mlle = list(range(len(labels_dr_mlle)))
y_mlle = f1_scores_mlle
z_mlle = accuracy_scores_mlle
fig, ax = plt.subplots(2, 1, figsize=(60, 30))
ax[0].plot(x_lle, y_lle, 'o-')
ax[0].plot(x_mlle, y_mlle, 'x-')
ax[0].axhline(y=f1_scores_raw, color='r', linestyle='-', linewidth=2)
ax[0].set_xticks(x_lle)
ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
ax[0].grid()
ax[0].legend(['lle', 'mlle', 'raw'], fontsize=20)
ax[0].set_title('F1-score', fontsize=30)
ax[0].tick_params(axis='y', which='major', labelsize=20)
ax[1].plot(x_lle, z_lle, 'o-')
ax[1].plot(x_mlle, z_mlle, 'x-')
ax[1].axhline(y=accuracy_scores_raw, color='r', linestyle='-', linewidth=2)
ax[1].set_xticks(x_mlle)
ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
ax[1].grid()
ax[1].legend(['lle', 'mlle', 'raw'], fontsize=20)
ax[1].set_title('Accuracy', fontsize=30)
ax[1].tick_params(axis='y', which='major', labelsize=20)
fig.savefig('./plots/f1_acc.png')
counter = 0
fig, ax = plt.subplots(3, len(nb), figsize=(120, 30))
sns.heatmap(cfm_raw, annot=True, ax=ax[0, 0])
ax[0, 0].set_title(states_raw, fontsize=20)
for i in range(len(nb)):
    sns.heatmap(cfm_lle[i], annot=True, ax=ax[1, i])
    ax[1, i].set_title(states_lle[i], fontsize=20)
for i in range(len(nb)):
    sns.heatmap(cfm_mlle[i], annot=True, ax=ax[2, i])
    ax[2, i].set_title(states_mlle[i], fontsize=20)
fig.savefig('./plots/cfm.png')


#
# methods = OrderedDict(); methods['RAW'] = []
# X_train_dict = OrderedDict(); X_train_dict['RAW'] = X_train
# X_test_dict = OrderedDict(); X_test_dict['RAW'] = X_test
# elapsed_dict = OrderedDict(); elapsed_dict['RAW'] = 0
#
#
#
# LLE = partial(manifold.LocallyLinearEmbedding,
#               eigen_solver='dense',
#               neighbors_algorithm='auto',
#               n_neighbors=21, # 21, 13
#               n_components=13,
#               random_state=617)
# methods['LLE'] = LLE(method="standard")
# start_time = time.time()
# X_train_dict['LLE'] = methods['LLE'].fit_transform(X_train)
# X_test_dict['LLE'] = methods['LLE'].transform(X_test)
# elapsed_time = time.time() - start_time
# elapsed_dict['LLE'] = elapsed_time
# print('LLE' + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# methods['MLLE'] = LLE(method="modified")
# start_time = time.time()
# X_train_dict['MLLE'] = methods['MLLE'].fit_transform(X_train)
# X_test_dict['MLLE'] = methods['MLLE'].transform(X_test)
# elapsed_time = time.time() - start_time
# elapsed_dict['MLLE'] = elapsed_time
# print('MLLE' + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
#
# # --- classification --- #
# states = []; elapsed = []; f1_scores = []; accuracy_scores = []; cfms = []
# clfs = OrderedDict()
#
#
# for method in methods:
#     for clf in clfs:
#         start_time = time.time()
#         clfs[clf].fit(X_train_dict[method], y_train)
#         predictions = clfs[clf].predict(X_test_dict[method])
#         elapsed_time = time.time() - start_time
#         states.append(method + ':' + clf + ':' + f'{elapsed_time:.2f}' + ':' + f'{elapsed_time + elapsed_dict[method]:.2f}')
#         print(states[-1])
#         elapsed.append(elapsed_time + elapsed_dict[method])
#         f1_scores.append(metrics.f1_score(y_test, predictions, average='weighted'))
#         accuracy_scores.append(metrics.accuracy_score(y_test, predictions))
#         cfms.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
# result = {'states': states, 'f1_scores': f1_scores, 'accuracy_scores': accuracy_scores, 'cfms': cfms, 'time': elapsed}
# result_df = pd.DataFrame(data=result)
# result_df.to_csv('./data/results.csv')
# ax = result_df.plot(x="states", y=["f1_scores", "accuracy_scores"], kind='bar', figsize=(30, 20))
# plt.show()
# fig, ax = plt.subplots(3, 3, figsize=(40, 40))
# counter = 0
# for i in range(len(methods)):
#     for j in range(len(clfs)):
#         sns.heatmap(cfms[counter], annot=True, ax=ax[i, j])
#         ax[i, j].set_title(states[counter], fontsize=20)
#         counter = counter + 1
# fig.savefig('./plots/all_02.png')
# print(accuracy_scores)
# print(f1_scores)
