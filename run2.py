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
from sklearn.svm import LinearSVC


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


# --- sampling and balancing --- #
# # spliting to train and test
y = data['hospital_death']
X = data.drop(['hospital_death'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=617)
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
# # sampling
over = SMOTE(sampling_strategy=death_proportion_spl*1.1, random_state=617)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=617)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_train_smp, y_train_smp = pipeline.fit_resample(X_train, y_train)
print(f'Train Sample: #data = {X_train_smp.shape[0]}, #features = {X_train_smp.shape[1]}')
nb_patients_smp = len(y_train_smp)
nb_survived_smp = len(y_train_smp[y_train_smp == 0])
nb_died_smp = len(y_train_smp[y_train_smp == 1])
death_proportion_smp = nb_died_smp / nb_patients_smp
print(f'Train Sample: #patients = {nb_patients_smp}, #survived = {nb_survived_smp}, #died = {nb_died_smp}, proportion = {death_proportion_smp:.3f}')


# --- dimensionality reduction --- #
methods = OrderedDict(); methods['RAW'] = []
X_train_dict = OrderedDict(); X_train_dict['RAW'] = X_train_smp
X_test_dict = OrderedDict(); X_test_dict['RAW'] = X_test
elapsed_dict = OrderedDict(); elapsed_dict['RAW'] = 0
LLE = partial(manifold.LocallyLinearEmbedding,
              eigen_solver='dense',
              neighbors_algorithm='auto',
              n_neighbors=21,
              n_components=13,
              random_state=617)
methods['LLE'] = LLE(method="standard")
start_time = time.time()
X_train_dict['LLE'] = methods['LLE'].fit_transform(X_train_smp)
X_test_dict['LLE'] = methods['LLE'].transform(X_test)
elapsed_time = time.time() - start_time
elapsed_dict['LLE'] = elapsed_time
print('LLE' + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
methods['MLLE'] = LLE(method="modified")
start_time = time.time()
X_train_dict['MLLE'] = methods['MLLE'].fit_transform(X_train_smp)
X_test_dict['MLLE'] = methods['MLLE'].transform(X_test)
elapsed_time = time.time() - start_time
elapsed_dict['MLLE'] = elapsed_time
print('MLLE' + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')

# --- classification --- #
states = []; elapsed = []; f1_scores = []; accuracy_scores = []; cfms = []
clfs = OrderedDict()
clfs['LR'] = LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=True, random_state=617, max_iter=2000)
clfs['SVM'] = LinearSVC(C=1.0, random_state=617, max_iter=5000)
i = 0
for method in methods:
    for clf in clfs:
        start_time = time.time()
        clfs[clf].fit(X_train_dict[method], y_train_smp)
        predictions = clfs[clf].predict(X_test_dict[method])
        elapsed_time = time.time() - start_time
        states.append(method + ':' + clf + ':' + f'{elapsed_time:.2f}')
        elapsed.append(elapsed_time + elapsed_dict[method])
        f1_scores.append(metrics.f1_score(y_test, predictions, average='weighted'))
        accuracy_scores.append(metrics.accuracy_score(y_test, predictions))
        cfms.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
        print(states[i] + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
        i = i + 1
result = {'states': states, 'f1_scores': f1_scores, 'accuracy_scores': accuracy_scores, 'cfms': cfms, 'time': elapsed}
result_df = pd.DataFrame(data=result)
result_df.to_csv('./data/results.csv')
ax = result_df.plot(x="states", y=["f1_scores", "accuracy_scores"], kind='bar', figsize=(30, 20))
# fig.savefig('./plots/all_01.png')
plt.show()
# plt.savefig('./plots/all_01.png')
fig, ax = plt.subplots(3, 2, figsize=(40, 40))
counter = 0
for i in range(len(methods)):
    for j in range(len(clfs)):
        sns.heatmap(cfms[counter], annot=True, ax=ax[i, j])
        ax[i, j].set_title(states[counter], fontsize=20)
        counter = counter + 1
fig.savefig('./plots/all_02.png')
print(accuracy_scores)
print(f1_scores)
