# --- modules --- #
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from collections import OrderedDict
from functools import partial
from itertools import combinations
import pandas as pd
import seaborn as sns       # sns.set()
import plotly.express as px
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture


# --- functions --- #
def plot_features(data, feature_name):
    if data[feature_name].value_counts().shape[0] > 20:
        plt.figure(figsize=(12, 8))
        sns.histplot(data=data[feature_name][data['hospital_death'] == 0], color='g',
                     label="Survived", kde=True, stat='density')
        sns.histplot(data=data[feature_name][data['hospital_death'] == 1], color='r',
                     label="Died", kde=True, stat='density')
        plt.legend(loc='upper right')
        plt.show()
    else:
        plt.figure(figsize=(14, 8))
        sns.countplot(x=feature_name, hue="hospital_death", data=data, palette='coolwarm')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.show()


def apply_eda(data, feature_name):
    sns.set(rc={'figure.figsize': (20, 20)})
    sns.heatmap(data.corr())
    plt.savefig('./plots/dependency.png')
    plot_features(data, feature_name)
    fig = px.histogram(data[['age', 'gender', 'hospital_death', 'bmi']], x="age", y="hospital_death", color="gender",
                       marginal="box", hover_data=data[['age', 'gender', 'hospital_death', 'bmi']].columns)
    fig.show()
    unpivot = pd.melt(data, data.describe().columns[0], data.describe().columns[1:])
    g = sns.FacetGrid(unpivot, col="variable", col_wrap=3, sharex=False, sharey=False)
    g.map(sns.kdeplot, "value")
    plt.savefig('./plots/all_distributions.png')


# --- data description --- #
# 'encounter_id': 'Unique identifier associated with a patient unit stay',
# 'patient_id': 'Unique identifier associated with a patient',
# 'hospital_id': 'Unique identifier associated with a hospital',
# 'age': 'The age of the patient on unit admission',
# 'bmi': 'The body mass index of the person on unit admission',
# 'elective_surgery': 'Whether the patient was admitted to the hospital for an elective surgical operation',
# 'ethnicity': 'The common national or cultural tradition which the person belongs to',
# 'gender': 'Sex of the patient',
# 'height': 'The height of the person on unit admission',
# 'icu_admit_source': 'The location of the patient prior to being admitted to the unit',
# 'icu_id': 'unique identifier for the unit to which the patient was admitted',
# 'icu_stay_type': 'unit type',
# 'icu_type': 'A classification which indicates the type of care the unit is capable of providing',
# 'pre_icu_los_days': 'The length of stay of the patient between hospital admission and unit admission',
# 'weight': 'The weight (body mass) of the person on unit admission',
# 'apache_2_diagnosis': 'The APACHE II diagnosis for the ICU admission',
# 'apache_3j_diagnosis': 'The APACHE III-J sub-diagnosis code which best describes the reason for the ICU admission',
# 'apache_post_operative': 'The APACHE operative status; 1 for post-operative, 0 for non-operative',
# 'arf_apache': 'Whether the patient had acute renal failure during the first 24 hours of their unit stay, defined as a 24 hour urine output <410ml, creatinine >=133 micromol/L and no chronic dialysis',
# 'gcs_eyes_apache': 'The eye opening component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score',
# 'gcs_motor_apache': 'The motor component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score',
# 'gcs_unable_apache': 'Whether the Glasgow Coma Scale was unable to be assessed due to patient sedation',
# 'gcs_verbal_apache': 'The verbal component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score',
# 'heart_rate_apache': 'The heart rate measured during the first 24 hours which results in the highest APACHE III score',
# 'intubated_apache': 'Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score',
# 'map_apache': 'The mean arterial pressure measured during the first 24 hours which results in the highest APACHE III score',
# 'resprate_apache': 'The respiratory rate measured during the first 24 hours which results in the highest APACHE III score',
# 'temp_apache': 'The temperature measured during the first 24 hours which results in the highest APACHE III score',
# 'ventilated_apache': 'Whether the patient was invasively ventilated at the time of the highest scoring arterial blood gas using the oxygenation scoring algorithm, including any mode of positive pressure ventilation delivered through a circuit attached to an endo-tracheal tube or tracheostomy',
# 'd1_diasbp_max': "The patient's highest diastolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured",
# 'd1_diasbp_min': "The patient's lowest diastolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured",
# 'd1_diasbp_noninvasive_max': "The patient's highest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured",
# 'd1_diasbp_noninvasive_min': "The patient's lowest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured",
# 'd1_heartrate_max': "The patient's highest heart rate during the first 24 hours of their unit stay",
# 'd1_heartrate_min': "The patient's lowest heart rate during the first 24 hours of their unit stay",
# 'd1_mbp_max': "The patient's highest mean blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured",
# 'd1_mbp_min': "The patient's lowest mean blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured",
# 'd1_mbp_noninvasive_max': "The patient's highest mean blood pressure during the first 24 hours of their unit stay, non-invasively measured",
# 'd1_mbp_noninvasive_min': "The patient's lowest mean blood pressure during the first 24 hours of their unit stay, non-invasively measured",
# 'd1_resprate_max': "The patient's highest respiratory rate during the first 24 hours of their unit stay",
# 'd1_resprate_min': "The patient's lowest respiratory rate during the first 24 hours of their unit stay",
# 'd1_spo2_max': "The patient's highest peripheral oxygen saturation during the first 24 hours of their unit stay",
# 'd1_spo2_min': "The patient's lowest peripheral oxygen saturation during the first 24 hours of their unit stay",
# 'd1_sysbp_max': "The patient's highest systolic blood pressure :uring the first 24 hours of their unit stay, either non-invasively or invasively measured",
# 'd1_sysbp_min': "The patient's lowest systolic blood pressure :uring the first 24 hours of their unit stay, either non-invasively or invasively measured",
# 'd1_sysbp_noninvasive_max': "The patient': highest systolic blood pressure during the first 24 hours of their unit stay, invasively measured",
# 'd1_sysbp_noninvasive_min': "The patient': lowest systolic blood pressure during the first 24 hours of their unit stay, invasively measured",
# 'd1_temp_max': "The patient:s highest core temperature during the first 24 hours of their unit stay, invasively measured",
# 'd1_temp_min': "The patient's lowest core temperature during the first 24 hours of their unit stay",
# 'h1_diasbp_max': "The patient's highest diastolic blood :ressure during the first hour of their unit stay, either non-invasively or invasively measured",
# 'h1_diasbp_min': "The patient's lowest diastolic blood :ressure during the first hour of their unit stay, either non-invasively or invasively measured",
# 'h1_diasbp_noninvasive_max': "The patient:s highest diastolic blood pressure during the first hour of their unit stay, invasively measured",
# 'h1_diasbp_noninvasive_min': "The patient:s lowest diastolic blood pressure during the first hour of their unit stay, invasively measured",
# 'h1_heartrate_max': "The patient's highest heart rate during the first hour of their unit stay",
# 'h1_heartrate_min': "The patient's lowest heart rate during the first hour of their unit stay",
# 'h1_mbp_max': "The patient's highest mean blood :ressure during the first hour of their unit stay, either non-invasively or invasively measured",
# 'h1_mbp_min': "The patient's lowest mean blood :ressure during the first hour of their unit stay, either non-invasively or invasively measured",
# 'h1_mbp_noninvasive_max': "The patient's highest mean blood pressure during the first hour of their unit stay, non-invasively measured",
# 'h1_mbp_noninvasive_min': "The patient's lowest mean blood pressure during the first hour of their unit stay, non-invasively measured",
# 'h1_resprate_max': "The patient's highest respiratory rate during the first hour of their unit stay",
# 'h1_resprate_min': "The patient's lowest respiratory rate during the first hour of their unit stay",
# 'h1_spo2_max': "The patient's highest peripheral oxygen saturation during the first hour of their unit stay",
# 'h1_spo2_min': "The patient's lowest peripheral oxygen saturation during the first hour of their unit stay",
# 'h1_sysbp_max': "The patient's highest systolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured",
# 'h1_sysbp_min': "The patient's lowest systolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured",
# 'h1_sysbp_noninvasive_max': "The patient's highest systolic blood pressure during the first hour of their unit stay, non-invasively measured",
# 'h1_sysbp_noninvasive_min': "The patient's lowest systolic blood pressure during the first hour of their unit stay, non-invasively measured",
# 'd1_glucose_max': "The highest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay",
# 'd1_glucose_min': "The lowest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay",
# 'd1_potassium_max': "The highest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay",
# 'd1_potassium_min': "The lowest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay",
# 'apache_4a_hospital_death_prob': "The APACHE IVa probabilistic prediction of in-hospital mortality for the patient which utilizes the APACHE III score and other covariates, including diagnosis.",
# 'apache_4a_icu_death_prob': "The APACHE IVa probabilistic prediction of in ICU mortality for the patient which utilizes the APACHE III score and other covariates, including diagnosis",
# 'aids': "Whether the patient has a definitive diagnosis of acquired immune deficiency syndrome (AIDS) (not HIV positive alone)",
# 'cirrhosis': "Whether the patient has a history of heavy alcohol use with portal hypertension and varices, other causes of cirrhosis with evidence of portal hypertension and varices, or biopsy proven cirrhosis. This comorbidity does not apply to patients with a functioning liver transplant.",
# 'diabetes_mellitus': "Whether the patient has been diagnosed with diabetes, either juvenile or adult onset, which requires medication.",
# 'hepatic_failure': "Whether the patient has cirrhosis and additional complications including jaundice and ascites, upper GI bleeding, hepatic encephalopathy, or coma.",
# 'immunosuppression': "Whether the patient has their immune system suppressed within six months prior to ICU admission for any of the following reasons; radiation therapy, chemotherapy, use of non-cytotoxic immunosuppressive drugs, high dose steroids (at least 0.3 mg/kg/day of methylprednisolone or equivalent for at least 6 months).",
# 'leukemia': "Whether the patient has been diagnosed with acute or chronic myelogenous leukemia, acute or chronic lymphocytic leukemia, or multiple myeloma.",
# 'lymphoma': "Whether the patient has been diagnosed with non-Hodgkin lymphoma.",
# 'solid_tumor_with_metastasis': "Whether the patient has been diagnosed with any solid tumor carcinoma (including malignant melanoma) which has evidence of metastasis.",
# 'apache_3j_bodysystem': "Admission diagnosis group for APACHE III",
# 'apache_2_bodysystem': "Admission diagnosis group for APACHE II",
# 'hospital_death': "Whether the patient died during this hospitalization"


# --- preprocessing --- #
data = pd.read_csv("./data/data.csv")

# remove empty feature
data.drop(['Unnamed: 83'], axis=1, inplace=True)

# # remove useless features
data.drop(['encounter_id', 'patient_id', 'hospital_id',
           'icu_admit_source', 'icu_id', 'icu_stay_type', 'icu_type',
           'pre_icu_los_days', 'apache_post_operative', 'gcs_unable_apache',
           'apache_3j_bodysystem', 'apache_2_bodysystem'
           ], axis=1, inplace=True)

# # remove rows with at least one null value in all features
data = data[data.isna().sum(axis=1) == 0]
#
# # Converting categorical values to numerical ones
for item in ['ethnicity', 'gender']:
    data[item] = data[item].astype('category')
    data[item+'_num'] = data[item].cat.codes
    data.drop(item, axis=1, inplace=True)
# # data.to_csv('./data/data_pp.csv', index=False)
# print(f'Original: #data = {data.shape[0]}, #features = {data.shape[1]}')
# nb_patients_org = len(data['hospital_death'])
# nb_survived_org = len(data[data['hospital_death'] == 0])
# nb_died_org = len(data[data['hospital_death'] == 1])
# death_proportion_org = nb_died_org / nb_patients_org
# print(f'Original: #patients = {nb_patients_org}, #survived = {nb_survived_org}, #died = {nb_died_org}, proportion = {death_proportion_org:.3f}')


# # --- sampling and balancing --- #
# # # spliting to train and test
# y = data['hospital_death']
# X = data.drop(['hospital_death'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=617)
# # # normalizing train and test
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(f'Train Split: #data = {X_train.shape[0]}, #features = {X_train.shape[1]}')
# nb_patients_spl = len(y_train)
# nb_survived_spl = len(y_train[y_train == 0])
# nb_died_spl = len(y_train[y_train == 1])
# death_proportion_spl = nb_died_spl / nb_patients_spl
# print(f'Train Split: #patients = {nb_patients_spl}, #survived = {nb_survived_spl}, #died = {nb_died_spl}, proportion = {death_proportion_spl:.3f}')
# # # sampling
# over = SMOTE(sampling_strategy=death_proportion_spl*1.1, random_state=617)
# under = RandomUnderSampler(sampling_strategy=0.8, random_state=617)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)
# X_train_smp, y_train_smp = pipeline.fit_resample(X_train, y_train)
# print(f'Train Sample: #data = {X_train_smp.shape[0]}, #features = {X_train_smp.shape[1]}')
# nb_patients_smp = len(y_train_smp)
# nb_survived_smp = len(y_train_smp[y_train_smp == 0])
# nb_died_smp = len(y_train_smp[y_train_smp == 1])
# death_proportion_smp = nb_died_smp / nb_patients_smp
# print(f'Train Sample: #patients = {nb_patients_smp}, #survived = {nb_survived_smp}, #died = {nb_died_smp}, proportion = {death_proportion_smp:.3f}')
#
#
# # --- dimensionality reduction --- #
# # # kpca
# # nb_components_kpca = 20
# # kpca = KernelPCA(n_components=nb_components_kpca, kernel='rbf')
# # X_train_kpca = kpca.fit_transform(X_train_smp)
# # fig, ax = plt.subplots(nb_components_kpca, nb_components_kpca, figsize=(70, 70))
# # for i in range(nb_components_kpca):
# #     for j in range(nb_components_kpca):
# #         if i == j:
# #             continue
# #         ax[i, j].scatter(X_train_kpca[y_train_smp == 0, i],
# #                          X_train_kpca[y_train_smp == 0, j], c='green', label='Survived')
# #         ax[i, j].scatter(X_train_kpca[y_train_smp == 1, i],
# #                          X_train_kpca[y_train_smp == 1, j], c='red', label='Died')
# #         ax[i, j].xaxis.set_major_formatter(NullFormatter())
# #         ax[i, j].yaxis.set_major_formatter(NullFormatter())
# #         ax[i, j].axis('tight')
# #         ax[i, j].legend()
# #         ax[i, j].set_xlabel(f'dim : {i+1}')
# #         ax[i, j].set_ylabel(f'dim : {j+1}')
# # fig.savefig('plots/kpca.png')
# # fig, ax = plt.subplots()
# # ax.plot(kpca.eigenvalues_, 'b', linewidth=2)
# # ax.set_xlabel('#Compeonents')
# # ax.set_ylabel('Eigenvalues')
# # fig.savefig('plots/kpca_eigenvalues.png')
# # eigen_cum = np.cumsum(kpca.eigenvalues_) / np.sum(kpca.eigenvalues_)
# # fig, ax = plt.subplots()
# # ax.plot(eigen_cum, 'b', linewidth=2)
# # ax.set_xlabel('#Compeonents')
# # ax.set_ylabel('Cumulative Eigenvalues')
# # fig.savefig('plots/kpca_eigenvalues_cum.png')
# nb_components_kpca = 10
# kpca = KernelPCA(n_components=nb_components_kpca, kernel='rbf')
# X_train_kpca = kpca.fit_transform(X_train_smp)
# X_test_kpca = kpca.transform(X_test)
#
# # # lle and mlle
# nb = np.arange(nb_components_kpca-5, nb_components_kpca+10, 1)
# # nb = np.arange(15, 25, 1)
# methods = OrderedDict()
# elapsed_dr_lle = OrderedDict()
# elapsed_dr_mlle = OrderedDict()
# X_train_dict_lle = OrderedDict()
# X_test_dict_lle = OrderedDict()
# X_train_dict_mlle = OrderedDict()
# X_test_dict_mlle = OrderedDict()
# labels_dr_lle = []
# labels_dr_mlle = []
# for i in range(len(nb)):
#     labels_dr_lle.append('LLE:' + str(nb[i]))
#     labels_dr_mlle.append('MLLE:' + str(nb[i]))
# LLE = partial(manifold.LocallyLinearEmbedding,
#               eigen_solver='dense',
#               neighbors_algorithm='auto',
#               n_neighbors=13,
#               random_state=617)
# for i in range(len(nb)):
#     methods[labels_dr_lle[i]] = LLE(n_neighbors=nb[i], method="standard")
#     start_time = time.time()
#     X_train_dict_lle[labels_dr_lle[i]] = methods[labels_dr_lle[i]].fit_transform(X_train_smp)
#     X_test_dict_lle[labels_dr_lle[i]] = methods[labels_dr_lle[i]].transform(X_test)
#     elapsed_time = time.time() - start_time
#     elapsed_dr_lle[labels_dr_lle[i]] = elapsed_time
#     print(labels_dr_lle[i] + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# for i in range(len(nb)):
#     methods[labels_dr_mlle[i]] = LLE(n_neighbors=nb[i], method="modified")
#     start_time = time.time()
#     X_train_dict_mlle[labels_dr_mlle[i]] = methods[labels_dr_mlle[i]].fit_transform(X_train_smp)
#     X_test_dict_mlle[labels_dr_mlle[i]] = methods[labels_dr_mlle[i]].transform(X_test)
#     elapsed_time = time.time() - start_time
#     elapsed_dr_mlle[labels_dr_mlle[i]] = elapsed_time
#     print(labels_dr_mlle[i] + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
#
# # --- classification --- #
# # # classifier = SVC(C=1.0, kernel='rbf', gamma='scale', probability=False, class_weight='balanced')
# # # classifier = GaussianProcessClassifier()
# # # classifier = GaussianNB()
# classifier = LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=True, random_state=617, max_iter=2000)
# # classifier = LinearSVC(C=1.0, random_state=617)
# # # raw
# start_time = time.time()
# classifier.fit(X_train_smp, y_train_smp)
# predictions = classifier.predict(X_test)
# elapsed_time_raw = time.time() - start_time
# states_raw = 'RAW:' + f'{elapsed_time_raw:.2f}'
# f1_scores_raw = metrics.f1_score(y_test, predictions, average='weighted')
# accuracy_scores_raw = metrics.accuracy_score(y_test, predictions)
# cfm_raw = metrics.confusion_matrix(y_test, predictions, normalize='true')
# print('RAW finished in ' + f'{elapsed_time_raw:.2f}' + ' s!')
# # # kpca
# start_time = time.time()
# classifier.fit(X_train_kpca, y_train_smp)
# predictions = classifier.predict(X_test_kpca)
# elapsed_time_kpca = time.time() - start_time
# states_kpca = 'KPCA:' + str(nb_components_kpca) + ':' + f'{elapsed_time_kpca:.2f}'
# f1_scores_kpca = metrics.f1_score(y_test, predictions, average='weighted')
# accuracy_scores_kpca = metrics.accuracy_score(y_test, predictions)
# cfm_kpca = metrics.confusion_matrix(y_test, predictions, normalize='true')
# print('KPCA:' + str(nb_components_kpca) + ' finished in ' + f'{elapsed_time_kpca:.2f}' + ' s!')
# # # lle
# elapsed_tot_lle = []; f1_scores_lle = []; accuracy_scores_lle = []; cfm_lle = []; states_lle = []
# for label in labels_dr_lle:
#     start_time = time.time()
#     classifier.fit(X_train_dict_lle[label], y_train_smp)
#     predictions = classifier.predict(X_test_dict_lle[label])
#     elapsed_time = time.time() - start_time
#     elapsed_tot_lle.append(elapsed_time+elapsed_dr_lle[label])
#     states_lle.append(label + ':' + f'{elapsed_tot_lle[-1]:.2f}')
#     f1_scores_lle.append(metrics.f1_score(y_test, predictions, average='weighted'))
#     accuracy_scores_lle.append(metrics.accuracy_score(y_test, predictions))
#     cfm_lle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
#     print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # # mlle
# elapsed_tot_mlle = []; f1_scores_mlle = []; accuracy_scores_mlle = []; cfm_mlle = []; states_mlle = []
# for label in labels_dr_mlle:
#     start_time = time.time()
#     classifier.fit(X_train_dict_mlle[label], y_train_smp)
#     predictions = classifier.predict(X_test_dict_mlle[label])
#     elapsed_time = time.time() - start_time
#     elapsed_tot_mlle.append(elapsed_time+elapsed_dr_mlle[label])
#     states_mlle.append(label + ':' + f'{elapsed_tot_mlle[-1]:.2f}')
#     f1_scores_mlle.append(metrics.f1_score(y_test, predictions, average='weighted'))
#     accuracy_scores_mlle.append(metrics.accuracy_score(y_test, predictions))
#     cfm_mlle.append(metrics.confusion_matrix(y_test, predictions, normalize='true'))
#     print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# # # plots
# x_lle = list(range(len(labels_dr_lle)))
# y_lle = f1_scores_lle
# z_lle = accuracy_scores_lle
# x_mlle = list(range(len(labels_dr_mlle)))
# y_mlle = f1_scores_mlle
# z_mlle = accuracy_scores_mlle
# fig, ax = plt.subplots(2, 1, figsize=(60, 30))
# ax[0].plot(x_lle, y_lle, 'o-')
# ax[0].plot(x_mlle, y_mlle, 'x-')
# ax[0].axhline(y=f1_scores_raw, color='r', linestyle='-', linewidth=2)
# ax[0].axhline(y=f1_scores_kpca, color='k', linestyle='-', linewidth=2)
# ax[0].set_xticks(x_lle)
# ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
# ax[0].grid()
# ax[0].legend(['lle', 'mlle', 'raw', 'kpca'], fontsize=20)
# ax[0].set_title('F1-score', fontsize=30)
# ax[0].tick_params(axis='y', which='major', labelsize=20)
# ax[1].plot(x_lle, z_lle, 'o-')
# ax[1].plot(x_mlle, z_mlle, 'x-')
# ax[1].axhline(y=accuracy_scores_raw, color='r', linestyle='-', linewidth=2)
# ax[1].axhline(y=accuracy_scores_kpca, color='k', linestyle='-', linewidth=2)
# ax[1].set_xticks(x_mlle)
# ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
# ax[1].grid()
# ax[1].legend(['lle', 'mlle', 'raw', 'kpca'], fontsize=20)
# ax[1].set_title('Accuracy', fontsize=30)
# ax[1].tick_params(axis='y', which='major', labelsize=20)
# fig.savefig('./plots/f1_acc.png')
# counter = 0
# fig, ax = plt.subplots(3, len(nb), figsize=(120, 30))
# sns.heatmap(cfm_raw, annot=True, ax=ax[0, 0])
# ax[0, 0].set_title(states_raw, fontsize=20)
# sns.heatmap(cfm_kpca, annot=True, ax=ax[0, 1])
# ax[0, 1].set_title(states_kpca, fontsize=20)
# for i in range(len(nb)):
#     sns.heatmap(cfm_lle[i], annot=True, ax=ax[1, i])
#     ax[1, i].set_title(states_lle[i], fontsize=20)
# for i in range(len(nb)):
#     sns.heatmap(cfm_mlle[i], annot=True, ax=ax[2, i])
#     ax[2, i].set_title(states_mlle[i], fontsize=20)
# fig.savefig('./plots/cfm.png')
#
