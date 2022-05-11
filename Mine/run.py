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
from sklearn.decomposition import KernelPCA
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import pandas as pd
import seaborn as sns
# sns.set()
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
           'pre_icu_los_days', 'apache_post_operative', 'gcs_unable_apache',
           'apache_3j_bodysystem', 'apache_2_bodysystem'
           ], axis=1, inplace=True)

# # remove rows with at least one null value in all features
# print("No. of rows with missing values:", data.isnull().any(axis=1).sum())
data = data[data.isna().sum(axis=1) == 0]
# print("No. of rows with missing values:", data.isnull().any(axis=1).sum())

# # correct typo of 'apache_2_bodysystem' "Undefined diagnosis"
# data["apache_2_bodysystem"].replace({"Undefined diagnoses": "Undefined Diagnoses"}, inplace=True)

# # plot some graphs
# apply_eda(data, 'age')

# # Converting categorical values to numerical ones
for item in ['ethnicity', 'gender']:
    data[item] = data[item].astype('category')
    data[item+'_num'] = data[item].cat.codes
    data.drop(item, axis=1, inplace=True)
data.to_csv("./data_preprocessed.csv", index=False)
# print(data.shape)
# print(data.nunique())
# print(data.info(verbose=True, show_counts=True))

# # extract a random sample of data (take care of balancing)
nb_patients = len(data['hospital_death'])
nb_survived = len(data[data['hospital_death'] == 0])
nb_died = len(data[data['hospital_death'] == 1])
# print(nb_patients, nb_survived, nb_died)
death_proportion = nb_died / nb_patients
data_survived = data.loc[(data["hospital_death"] == 0)]
data_dead = data.loc[(data["hospital_death"] == 1)]
nb_samples = 10000
part1 = data_dead.sample(int(nb_samples * death_proportion), random_state=42)
part2 = data_survived.sample(nb_samples - int(nb_samples * death_proportion), random_state=42)
data_sampled = pd.concat([part1, part2])
data_sampled.to_csv('./data_sampled.csv', index=False)
# print(data_sampled.shape)

# --- deriving training and testing sets and normalizing (scaling them) --- #
y = data_sampled['hospital_death']
X = data_sampled.drop(['hospital_death'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# The stratify parameter makes a split so that the proportion of values in the sample produced will be the same as
# the proportion of values provided to parameter stratify.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
data_train_test = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)])
data_train_test.to_csv('./data_train_test.csv', index=False)
# print(data_train_test.shape)

nb_components = 35
nb_neighbors = np.arange(nb_components+2, 2*nb_components+1, 2)
methods = OrderedDict()
elapsed_dr_lle = OrderedDict()
elapsed_dr_mlle = OrderedDict()
X_train_dict_lle = OrderedDict()
X_test_dict_lle = OrderedDict()
X_train_dict_mlle = OrderedDict()
X_test_dict_mlle = OrderedDict()
labels_dr_lle = []
labels_dr_mlle = []
for i in range(len(nb_neighbors)):
    labels_dr_lle.append('LLE:' + str(nb_neighbors[i]))
    labels_dr_mlle.append('MLLE:' + str(nb_neighbors[i]))
LLE = partial(manifold.LocallyLinearEmbedding,
              eigen_solver='dense',
              neighbors_algorithm='auto',
              n_components=nb_components,
              random_state=42)
for label in labels_dr_lle:
    methods[label] = LLE(n_neighbors=int(label.split(':')[1]), method="standard")
    start_time = time.time()
    X_train_dict_lle[label] = methods[label].fit_transform(X_train)
    X_test_dict_lle[label] = methods[label].transform(X_test)
    elapsed_time = time.time() - start_time
    elapsed_dr_lle[label] = elapsed_time
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
for label in labels_dr_mlle:
    methods[label] = LLE(n_neighbors=int(label.split(':')[1]), method="modified")
    start_time = time.time()
    X_train_dict_mlle[label] = methods[label].fit_transform(X_train)
    X_test_dict_mlle[label] = methods[label].transform(X_test)
    elapsed_time = time.time() - start_time
    elapsed_dr_mlle[label] = elapsed_time
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')

# # classificatiom
classifier = SVC(C=1.0, kernel='rbf', gamma='scale', probability=False, class_weight='balanced')
elapsed_tot_lle = []
f1_scores_lle = []
accuracy_scores_lle = []
elapsed_tot_mlle = []
f1_scores_mlle = []
accuracy_scores_mlle = []
states_lle = []
states_mlle = []
# # main loop
for label in labels_dr_lle:
    start_time = time.time()
    classifier.fit(X_train_dict_lle[label], y_train)
    predictions = classifier.predict(X_test_dict_lle[label])
    elapsed_time = time.time() - start_time
    elapsed_tot_lle.append(elapsed_time+elapsed_dr_lle[label])
    states_lle.append(label + ':' + f'{elapsed_tot_lle[-1]:.2f}')
    f1_scores_lle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_lle.append(metrics.accuracy_score(y_test, predictions))
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
for label in labels_dr_mlle:
    start_time = time.time()
    classifier.fit(X_train_dict_mlle[label], y_train)
    predictions = classifier.predict(X_test_dict_mlle[label])
    elapsed_time = time.time() - start_time
    elapsed_tot_mlle.append(elapsed_time+elapsed_dr_mlle[label])
    states_mlle.append(label + ':' + f'{elapsed_tot_mlle[-1]:.2f}')
    f1_scores_mlle.append(metrics.f1_score(y_test, predictions, average='weighted'))
    accuracy_scores_mlle.append(metrics.accuracy_score(y_test, predictions))
    print(label + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
result_lle = {'states_lle': states_lle, 'f1_scores_lle': f1_scores_lle, 'accuracy_scores_lle': accuracy_scores_lle, 'time_lle': elapsed_tot_lle}
result_lle_df = pd.DataFrame(data=result_lle)
result_mlle = {'states_mlle': states_mlle, 'f1_scores_mlle': f1_scores_mlle, 'accuracy_scores_mlle': accuracy_scores_mlle, 'time_mlle': elapsed_tot_mlle}
result_mlle_df = pd.DataFrame(data=result_mlle)
# # set-up figures
x_lle = list(range(len(labels_dr_lle)))
y_lle = result_lle_df['f1_scores_lle']
z_lle = result_lle_df['accuracy_scores_lle']
x_mlle = list(range(len(labels_dr_mlle)))
y_mlle = result_mlle_df['f1_scores_mlle']
z_mlle = result_mlle_df['accuracy_scores_mlle']
fig, ax = plt.subplots(2, 1, figsize=(60, 30))
ax[0].plot(x_lle, y_lle, 'o-')
ax[0].plot(x_lle, z_lle, 'x-')
ax[0].set_xticks(x_lle)
ax[0].set_xticklabels(states_lle, rotation=45, fontsize=20)
ax[0].grid()
ax[1].plot(x_mlle, y_mlle, 'o-')
ax[1].plot(x_mlle, z_mlle, 'x-')
ax[1].set_xticks(x_mlle)
ax[1].set_xticklabels(states_mlle, rotation=45, fontsize=20)
ax[1].grid()
fig.savefig('./result_dr.png')


