# --- modules --- #
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from collections import OrderedDict
from functools import partial
from itertools import combinations
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
import pandas as pd
import seaborn as sns
sns.set()
import plotly.express as px
import numpy as np


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
    plt.savefig('dependency.png')
    plot_features(data, feature_name)
    fig = px.histogram(data[['age', 'gender', 'hospital_death', 'bmi']], x="age", y="hospital_death", color="gender",
                       marginal="box", hover_data=data[['age', 'gender', 'hospital_death', 'bmi']].columns)
    fig.show()
    unpivot = pd.melt(data, data.describe().columns[0], data.describe().columns[1:])
    g = sns.FacetGrid(unpivot, col="variable", col_wrap=3, sharex=False, sharey=False)
    g.map(sns.kdeplot, "value")
    plt.savefig('all_distributions.png')

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
data = pd.read_csv("./dataset.csv")

# # remove empty feature
data.drop(['Unnamed: 83'], axis=1, inplace=True)

# # remove useless features
data.drop(['encounter_id', 'patient_id', 'hospital_id',
           'icu_admit_source', 'icu_id', 'icu_stay_type', 'icu_type',
           'pre_icu_los_days', 'apache_post_operative', 'gcs_unable_apache'
           ], axis=1, inplace=True)

# # remove rows with at least one null value in all features
# print("No. of rows with missing values:", data.isnull().any(axis=1).sum())
data = data[data.isna().sum(axis=1) == 0]
# print("No. of rows with missing values:", data.isnull().any(axis=1).sum())

# # correct typo of 'apache_2_bodysystem' "Undefined diagnosis"
data["apache_2_bodysystem"].replace({"Undefined diagnoses": "Undefined Diagnoses"}, inplace=True)

# # plot some graphs
# apply_eda(data, 'age')

# # One hot encoding of string data
# print(data.shape)
for (columnName, columnData) in data.iteritems():
    if columnData.dtype == "object":
        one_hot = pd.get_dummies(columnData, prefix=columnName)
        data.drop([columnName], axis=1, inplace=True)
        data = data.join(one_hot)
# data.to_csv("./dataset_preprocessed.csv", index=False)
# print(data.shape)
# print(data.nunique())
# print(data.info(verbose=True, show_counts=True))

# # extract a random sample of data (take care of balancing)
data = data.sample(frac=1).reset_index(drop=True)   # shuffle
nb_patients = len(data['hospital_death'])
nb_survived = len(data[data['hospital_death'] == 0])
nb_died = len(data[data['hospital_death'] == 1])
# print(nb_patients, nb_survived, nb_died)
death_proportion = nb_died / nb_patients
data_survived = data.loc[(data["hospital_death"] == 0)]
data_dead = data.loc[(data["hospital_death"] == 1)]
nb_samples = 7000
part1 = data_dead.sample(int(nb_samples * death_proportion))
part2 = data_survived.sample(nb_samples - int(nb_samples * death_proportion))
data_sampled = pd.concat([part1, part2])

# --- deriving training and testing sets and normalizing (scaling them) --- #
y = data_sampled['hospital_death']
X = data_sampled.drop(['hospital_death'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# The stratify parameter makes a split so that the proportion of values in the sample produced will be the same as
# the proportion of values provided to parameter stratify.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --- dimensionality reduction --- #
LLE = partial(manifold.LocallyLinearEmbedding,
              eigen_solver='dense',
              neighbors_algorithm='auto')
nb_neighbors = 20
nb_components = 15
methods = OrderedDict()
methods['PCA'] = PCA(n_components=40)
methods['LLE'] = LLE(n_neighbors=nb_neighbors, n_components=nb_components, method="standard")
methods['MLLE'] = LLE(n_neighbors=nb_neighbors, n_components=nb_components, method="modified")
elapsed_dr = OrderedDict()
X_train_dict = OrderedDict()
X_test_dict = OrderedDict()
labels_dr = ['PCA', 'LLE', 'MLLE']
for label in labels_dr:
    start_time = time.time()
    X_train_dict[label] = methods[label].fit_transform(X_train)
    X_test_dict[label] = methods[label].transform(X_test)
    elapsed_time = time.time() - start_time
    elapsed_dr[label] = elapsed_time
    print(label + ' finished in ' + str(elapsed_time) + ' s!')
# # set-up lle vs mlle figure
list_comb = list(range(nb_components))
list_comb = list(combinations(list_comb, 2))
nb_pairs = len(list_comb)
fig, axs = plt.subplots(nb_pairs, 2, squeeze=False, figsize=(35, 18))
fig.suptitle('Manifold Learning with %i neighbors and %i embeddings' % (nb_neighbors, nb_components), fontsize=14)
for m, label in enumerate(labels_dr):
    if label == 'PCA':
        continue
    train = X_train_dict[label]
    n = m - 1
    for (l, x) in enumerate(list_comb):
        axs[l, n].scatter(train[y_train == 0, x[0]],
                          train[y_train == 0, x[1]], c='green', label='Survived')
        axs[l, n].scatter(train[y_train == 1, x[0]],
                          train[y_train == 1, x[1]], c='red', label='Died')
        if l == 0:
            axs[l, n].set_title('%s (%.2g sec)' % (label, elapsed_dr[label]))
        axs[l, n].xaxis.set_major_formatter(NullFormatter())
        axs[l, n].yaxis.set_major_formatter(NullFormatter())
        axs[l, n].axis('tight')
        axs[l, n].legend()
        axs[l, n].set_xlabel(f'dim : {x[0]}')
        axs[l, n].set_ylabel(f'dim : {x[1]}')
fig.savefig('LLE_MLLE.png')

# --- set-up classification --- #
classifiers = OrderedDict()
classifiers['Linear_SVM'] = LinearSVC(penalty='l2', loss='hinge', dual=True, C=1.0,
                                      fit_intercept=True, intercept_scaling=1,
                                      class_weight='balanced', max_iter=1000)
classifiers['RBF_SVM'] = SVC(C=1.0, kernel='rbf', gamma='scale', probability=False, class_weight='balanced')
labels_clf = ['Linear_SVM', 'RBF_SVM']
elapsed_tot = []
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []
cms = []
rocs = []
states = []
# # main loop
for label_clf in labels_clf:
    for label_dr in labels_dr:
        states.append(label_clf+':'+label_dr)
        start_time = time.time()
        classifiers[label_clf].fit(X_train_dict[label_dr], y_train)
        predictions = classifiers[label_clf].predict(X_test_dict[label_dr])
        elapsed_time = time.time() - start_time
        elapsed_tot.append(elapsed_time)
        precision_scores.append(metrics.precision_score(y_test, predictions))
        recall_scores.append(metrics.recall_score(y_test, predictions))
        f1_scores.append(metrics.f1_score(y_test, predictions))
        accuracy_scores.append(metrics.accuracy_score(y_test, predictions))
        cms.append(metrics.confusion_matrix(y_test, predictions))
        rocs.append(metrics.roc_curve(y_test, predictions))
        print(label_clf + ':' + label_dr + ' finished in ' + str(elapsed_time) + ' s!')
result = {'states': states, 'precision_scores': precision_scores, 'recall_scores': recall_scores,
          'f1_scores': f1_scores, 'accuracy_scores': accuracy_scores, 'time': elapsed_tot, 'cms': cms, 'rocs': rocs}
result_df = pd.DataFrame(data=result)
result_df.to_csv('./results.csv')
# # set-up figures
result_df_reduced = result_df[['precision_scores', 'recall_scores', 'f1_scores', 'accuracy_scores', 'time']]
ax = result_df.plot(kind='bar', figsize=(40, 20))
ax.set_xticklabels(states, rotation=45)
plt.savefig('results.png')
fig, axs = plt.subplots(len(labels_clf), len(labels_dr), squeeze=False, figsize=(40, 20))
counter = 0
for i in range(len(labels_clf)):
    for j in range(len(labels_dr)):
        sns.heatmap(cms[counter], annot=True, ax=axs[i, j])
        axs[i, j].set_title(states[counter])
        counter = counter + 1
fig.savefig('confusion_matrix.png')
fig, axs = plt.subplots(len(labels_clf), len(labels_dr), squeeze=False, figsize=(40, 20))
counter = 0
for i in range(len(labels_clf)):
    for j in range(len(labels_dr)):
        fpr, tpr, _ = rocs[counter]
        axs[i, j].plot(fpr, tpr)
        axs[i, j].set_title(states[counter])
        counter = counter + 1
fig.savefig('roc_curves.png')

# --- dummy --- #
# # 01
# print(X.info(verbose=True, show_counts=True))
# print(Y.info(verbose=True, show_counts=True))
# print(sampled_data.info(verbose=True, show_counts=True))
# print(death_proportion)
# # 02
# pca = PCA()
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
# explained_variance_s = pca.explained_variance_ratio_
# plt.figure(figsize=(10, 8))
# plt.plot(explained_variance_s)
# plt.show()
# nb_components = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97]
# for i in nb_components:
#      pca = PCA(n_components=i)
#      X_train_pca = pca.fit_transform(X_train)
#      print('Total Explained Variance Ratio using {} components = {}%'.format(i, round(np.sum(pca.explained_variance_ratio_)*100, 2)))
# # 03
# result_df.insert(loc=0, column='Classifier', value=labels)
