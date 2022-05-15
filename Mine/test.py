# # correct typo of 'apache_2_bodysystem' "Undefined diagnosis"
# data["apache_2_bodysystem"].replace({"Undefined diagnoses": "Undefined Diagnoses"}, inplace=True)

# print("No. of rows with missing values:", data.isnull().any(axis=1).sum())

# # plot some graphs
# apply_eda(data, 'age')

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










