# # correct typo of 'apache_2_bodysystem' "Undefined diagnosis"
# data["apache_2_bodysystem"].replace({"Undefined diagnoses": "Undefined Diagnoses"}, inplace=True)

# print("No. of rows with missing values:", data.isnull().any(axis=1).sum())

# print(data.nunique())
# print(data.info(verbose=True, show_counts=True))

# result_lle = {'states_lle': states_lle, 'f1_scores_lle': f1_scores_lle, 'accuracy_scores_lle': accuracy_scores_lle, 'time_lle': elapsed_tot_lle}
# result_lle_df = pd.DataFrame(data=result_lle)
# result_mlle = {'states_mlle': states_mlle, 'f1_scores_mlle': f1_scores_mlle, 'accuracy_scores_mlle': accuracy_scores_mlle, 'time_mlle': elapsed_tot_mlle}
# result_mlle_df = pd.DataFrame(data=result_mlle)

# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
# print(classification_report(KNN_prediction, y_test))

# Classification Accuracy is the simplest out of all the methods of evaluating the accuracy,
# and the most commonly used. Classification accuracy is simply the number of correct predictions divided by all
# predictions or a ratio of correct predictions to total predictions. While it can give you a quick idea of how your
# classifier is performing, it is best used when the number of observations/examples in each class is
# roughly equivalent. Because this doesn't happen very often, you're probably better off using another metric.

# Logarithmic Loss, or LogLoss, essentially evaluates how confident the classifier is about its predictions.
# LogLoss returns probabilities for membership of an example in a given class, summing them together to give
# a representation of the classifier's general confidence.

# The value for predictions runs from 1 to 0, with 1 being completely confident and 0 being no confidence.
# The loss, or overall lack of confidence, is returned as a negative number with 0 representing a perfect classifier,
# so smaller values are better.

# Area Under ROC Curve (AUC) is a metric used only for binary classification problems.The area under the curve
# represents the model's ability to properly discriminate between negative and positive examples, between
# one class or another.

# The classification report is a Scikit - Learn built in metric created especially for classification problems.
# Using the classification report can give you a quick intuition of how your model is performing.
# Recall pits the number of examples your model labeled as Class A (some given class ) against the total number of
# examples of Class A, and this is represented in the report.
# The report also returns prediction and f1-score. Precision is the percentage of examples your model labeled as
# Class A which actually belonged to Class A (true positives against false positives),
# and f1-score is an average of precision and recall.

# The stratify parameter makes a split so that the proportion of values in the sample produced will be the same as
# the proportion of values provided to parameter stratify.

# --- sampling --- #
# # extract a random sample of data (take care of balancing)
# nb_patients_org = len(data['hospital_death'])
# nb_survived_org = len(data[data['hospital_death'] == 0])
# nb_died_org = len(data[data['hospital_death'] == 1])
# print(f'Original: #patients = {nb_patients_org}, #survived = {nb_survived_org}, #died = {nb_died_org}')
# death_proportion = nb_died_org / nb_patients_org
# data_survived = data.loc[(data["hospital_death"] == 0)]
# data_dead = data.loc[(data["hospital_death"] == 1)]
# nb_samples = 8000
# part1 = data_dead.sample(int(nb_samples * death_proportion), random_state=42)
# part2 = data_survived.sample(nb_samples - int(nb_samples * death_proportion), random_state=42)
# data_sp = pd.concat([part1, part2], ignore_index=True)
# # data_sp.to_csv('./data/data_sp.csv', index=False)
# print(f'Sampled: #data = {data_sp.shape[0]}, #features = {data_sp.shape[1]}')
# nb_patients_sp = len(data_sp['hospital_death'])
# nb_survived_sp = len(data_sp[data_sp['hospital_death'] == 0])
# nb_died_sp = len(data_sp[data_sp['hospital_death'] == 1])
# print(f'Sampled: #patients = {nb_patients_sp}, #survived = {nb_survived_sp}, #died = {nb_died_sp}')

# --- deriving training and testing sets and normalizing (scaling them) --- #
# y = data_sp['hospital_death']
# X = data_sp.drop(['hospital_death'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_tt_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)
# y_tt_df = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], ignore_index=True)
# data_tt = pd.concat([X_tt_df, y_tt_df], axis=1, ignore_index=True)
# # data_tt.to_csv('./data/data_tt.csv', index=False)
# print(f'TrainTest: #data = {data_tt.shape[0]}, #features = {data_tt.shape[1]}')
# nb_patients_tt = len(data_tt.iloc[:, -1])
# nb_survived_tt = len(data_tt[data_tt.iloc[:, -1] == 0])
# nb_died_tt = len(data_tt[data_tt.iloc[:, -1] == 1])
# print(f'TrainTest: #patients = {nb_patients_tt}, #survived = {nb_survived_tt}, #died = {nb_died_tt}')

# # pca
# nb_components_pca = list(range(2, X_train.shape[1], 2))
# for i in nb_components_pca:
#      pca = PCA(n_components=i)
#      X_train_pca = pca.fit_transform(X_train_smp)
#      print('Total Explained Variance Ratio using {} components = {}%'.format(i, round(np.sum(pca.explained_variance_ratio_)*100, 2)))
# nb_components_pca = 34
# pca = PCA(n_components=nb_components_pca)
# X_train_pca = pca.fit_transform(X_train)
# fig, ax = plt.subplots(nb_components_pca, nb_components_pca, figsize=(70, 70))
# for i in range(nb_components_pca):
#     for j in range(nb_components_pca):
#         if i == j:
#             continue
#         ax[i, j].scatter(X_train_pca[y_train == 0, i],
#                          X_train_pca[y_train == 0, j], c='green', label='Survived')
#         ax[i, j].scatter(X_train_pca[y_train == 1, i],
#                          X_train_pca[y_train == 1, j], c='red', label='Died')
#         ax[i, j].xaxis.set_major_formatter(NullFormatter())
#         ax[i, j].yaxis.set_major_formatter(NullFormatter())
#         ax[i, j].axis('tight')
#         ax[i, j].legend()
#         ax[i, j].set_xlabel(f'dim : {i+1}')
#         ax[i, j].set_ylabel(f'dim : {j+1}')
# fig.savefig('plots/pca.png')



# # --- picking up single values --- #
# LLE = partial(manifold.LocallyLinearEmbedding,
#               eigen_solver='dense',
#               neighbors_algorithm='auto',
#               n_components=5,
#               n_neighbors=10,
#               random_state=42)
# labels_dr_lle = 'LLE:' + str(5) + ':' + str(10)
# labels_dr_mlle = 'MLLE:' + str(5) + ':' + str(10)
# lle = LLE(method="standard")
# start_time = time.time()
# X_train_lle = lle.fit_transform(X_train)
# X_test_lle = lle.transform(X_test)
# elapsed_time = time.time() - start_time
# elapsed_dr_lle = elapsed_time
# print(labels_dr_lle + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# mlle = LLE(method="modified")
# start_time = time.time()
# X_train_mlle = mlle.fit_transform(X_train)
# X_test_mlle = mlle.transform(X_test)
# elapsed_time = time.time() - start_time
# elapsed_dr_mlle = elapsed_time
# print(labels_dr_mlle + ' finished in ' + f'{elapsed_time:.2f}' + ' s!')
# classifier = LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=True, class_weight='balanced', random_state=42)
# start_time = time.time()
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
# elapsed_time_raw = time.time() - start_time
# states_raw = 'RAW:' + f'{elapsed_time_raw:.2f}'
# f1_scores_raw = metrics.f1_score(y_test, predictions, average='weighted')
# accuracy_scores_raw = metrics.accuracy_score(y_test, predictions)
# cfm_raw = metrics.confusion_matrix(y_test, predictions, normalize='true')
# print('RAW finished in ' + f'{elapsed_time_raw:.2f}' + ' s!')
# start_time = time.time()
# classifier.fit(X_train_kpca, y_train)
# predictions = classifier.predict(X_test_kpca)
# elapsed_time_kpca = time.time() - start_time
# states_kpca = 'KPCA:' + str(nb_components_kpca) + ':' + f'{elapsed_time_kpca:.2f}'
# f1_scores_kpca = metrics.f1_score(y_test, predictions, average='weighted')
# accuracy_scores_kpca = metrics.accuracy_score(y_test, predictions)
# cfm_kpca = metrics.confusion_matrix(y_test, predictions, normalize='true')
# print('KPCA:' + str(nb_components_kpca) + ' finished in ' + f'{elapsed_time_kpca:.2f}' + ' s!')
# start_time = time.time()
# classifier.fit(X_train_lle, y_train)
# predictions = classifier.predict(X_test_lle)
# elapsed_time_lle = time.time() - start_time
# states_lle = labels_dr_lle + ':' + f'{elapsed_time_lle:.2f}'
# f1_scores_lle = metrics.f1_score(y_test, predictions, average='weighted')
# accuracy_scores_lle = metrics.accuracy_score(y_test, predictions)
# cfm_lle = metrics.confusion_matrix(y_test, predictions, normalize='true')
# print(labels_dr_lle + ' finished in ' + f'{elapsed_time_lle:.2f}' + ' s!')
# start_time = time.time()
# classifier.fit(X_train_mlle, y_train)
# predictions = classifier.predict(X_test_mlle)
# elapsed_time_mlle = time.time() - start_time
# states_mlle = labels_dr_mlle + ':' + f'{elapsed_time_mlle:.2f}'
# f1_scores_mlle = metrics.f1_score(y_test, predictions, average='weighted')
# accuracy_scores_mlle = metrics.accuracy_score(y_test, predictions)
# cfm_mlle = metrics.confusion_matrix(y_test, predictions, normalize='true')
# print(labels_dr_mlle + ' finished in ' + f'{elapsed_time_mlle:.2f}' + ' s!')
# y_lle = f1_scores_lle
# z_lle = accuracy_scores_lle
# y_mlle = f1_scores_mlle
# z_mlle = accuracy_scores_mlle
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# ax[0].axhline(y=y_lle, color='b', linestyle='-', linewidth=1)
# ax[0].axhline(y=y_mlle, color='g', linestyle='-', linewidth=1)
# ax[0].axhline(y=f1_scores_raw, color='r', linestyle='-', linewidth=1)
# ax[0].axhline(y=f1_scores_kpca, color='k', linestyle='-', linewidth=1)
# ax[0].grid()
# ax[0].legend(['lle', 'mlle', 'raw', 'kpca'], fontsize=20)
# ax[0].set_title('F1-score', fontsize=30)
# ax[0].tick_params(axis='y', which='major', labelsize=20)
# ax[1].axhline(y=z_lle, color='b', linestyle='-', linewidth=1)
# ax[1].axhline(y=z_mlle, color='g', linestyle='-', linewidth=1)
# ax[1].axhline(y=accuracy_scores_raw, color='r', linestyle='-', linewidth=1)
# ax[1].axhline(y=accuracy_scores_kpca, color='k', linestyle='-', linewidth=1)
# ax[1].grid()
# ax[1].legend(['lle', 'mlle', 'raw', 'kpca'], fontsize=20)
# ax[1].set_title('Accuracy', fontsize=30)
# ax[1].tick_params(axis='y', which='major', labelsize=20)
# fig.savefig('./plots/f1_acc.png')
# fig, ax = plt.subplots(2, 2, figsize=(30, 30))
# sns.heatmap(cfm_raw, annot=True, ax=ax[0, 0])
# ax[0, 0].set_title(states_raw, fontsize=20)
# sns.heatmap(cfm_kpca, annot=True, ax=ax[0, 1])
# ax[0, 1].set_title(states_kpca, fontsize=20)
# sns.heatmap(cfm_lle, annot=True, ax=ax[1, 0])
# ax[1, 0].set_title(states_lle[0], fontsize=20)
# sns.heatmap(cfm_mlle, annot=True, ax=ax[1, 1])
# ax[1, 1].set_title(states_mlle[0], fontsize=20)
# fig.savefig('./plots/cfm.png')




