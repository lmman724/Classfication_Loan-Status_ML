import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

print("All library was imported")


df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv")
df.head()


df = df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis = "columns")


print("Columns and feature in dataset:", df.shape)


df["effective_date"]= pd.to_datetime(df["effective_date"])
df["due_date"] = pd.to_datetime(df["due_date"])


df.head()


df["day_of_name"] = df["effective_date"].dt.day_name()


df["day_of_name_number"] = df["effective_date"].dt.dayofweek


df.head()


df["loan_status"].value_counts()


df["Principal"].value_counts()


df["terms"].value_counts()


df["education"].value_counts()


df["Gender"].value_counts()


df_date_name = df[["day_of_name", "loan_status"]]
df_groupby_date_name = (df_date_name.groupby(["day_of_name"])["loan_status"]
                        .value_counts(normalize=True)
                        .rename("percentage")
                        .reset_index()
                        .sort_values("day_of_name"))
df_groupby_date_name


sns.countplot(x= "day_of_name", hue = "loan_status", data = df_date_name)


df["weekend"] = df["day_of_name_number"].apply(lambda x: 1 if (x> 3) else 0)


df.head()


df_gender = df[["Gender", "loan_status"]]
df_gender_groupby = (df_gender.groupby(["Gender"])["loan_status"]
                     .value_counts(normalize=True))
                     
                     
df_gender_groupby.head()


sns.countplot(x = "Gender", hue = "loan_status", data = df_gender)


df_education = df[["education","loan_status"]]


sns.countplot(x = "education", hue = "loan_status", data = df_education)


df_groupby_edu = (df_education.groupby(["education"])['loan_status']
                  .value_counts(normalize = True))
df_groupby_edu
                  


df_terms = df[["terms", "loan_status"]]


sns.countplot(x = "terms", hue = "loan_status", data = df_terms)


df_groupby_terms = df_terms.groupby(["terms"])["loan_status"].value_counts(normalize = True)
df_groupby_terms


df_terms_education = df[["terms", "education"]]


sns.countplot(x = "terms", hue = "education", data = df_terms_education)


df_terms_education_groupby = df_terms_education.groupby(["terms"])["education"].value_counts(normalize = True)
df_terms_education_groupby


df_terms_gender = df[["terms", "Gender"]]


sns.countplot(x = "terms", hue = "Gender", data = df_terms_gender)


df["Gender"] = df["Gender"].replace(to_replace = ["male", "female"], value = [0, 1])
df["Gender"]


df["loan_status"] = df["loan_status"].replace(to_replace = ["PAIDOFF", "COLLECTION"], value = [0, 1])


df["education"] = df["education"].replace(to_replace = ["High School or Below", "college", "Bechalor", "Master or Above"]
                                          , value = [0, 1, 2, 3])


df.head()


y = df["loan_status"]
y.head()


x = df[["Principal", "terms", "age", "education", "Gender", "weekend"]]
x.head()


x = x.to_numpy()


from sklearn.preprocessing import StandardScaler

sta = preprocessing.StandardScaler()
sta.fit(x)
x = sta.transform(x)


x


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


print("The shape of x_train and x_test:", x_train.shape, x_test.shape)
print("The shape of y_train and y_test:", y_train.shape, y_test.shape)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


k_nearest = 10

accuracy_knearest = np.zeros(k_nearest - 1)

for n in range(1, k_nearest):
    #train and predict model
    KNN = KNeighborsClassifier(n_neighbors = n)
    kneighbor_model = KNN.fit(x_train, y_train)
    y_pred_KNN_demo = kneighbor_model.predict(x_test)
    accuracy_knearest[n - 1] = metrics.accuracy_score(y_test, y_pred_KNN_demo)
    
accuracy_knearest


plt.plot(range(1, k_nearest), accuracy_knearest, "g")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy score")
plt.tight_layout()
plt.show()


k = 4
KNN_offical = KNeighborsClassifier(n_neighbors = 4)
KNN_model_official = KNN_offical.fit(x_train, y_train)

y_pred_KNN = KNN_model_official.predict(x_test)


f1_score_KNN = f1_score(y_test,y_pred_KNN)
jaccard_score_KNN = jaccard_score(y_test, y_pred_KNN)
log_loss_KNN = log_loss(y_test, y_pred_KNN)


print("F1_score value in KNN model:", round(f1_score_KNN, 4))
print("Jaccard score value in KNN model:", round(jaccard_score_KNN, 4))
print("Log loss value in KNN model:", round(log_loss_KNN, 4))


confusion_matrix_KNN = plot_confusion_matrix(KNN_model_official,x_test, y_test)


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion = "gini", 
                                       max_depth = 6, random_state= 0)
tree_model = decision_tree.fit(x_train, y_train)

y_pred_tree = tree_model.predict(x_test)


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(tree_model.fit(x_train, y_train))


f1_score_tree = f1_score(y_test, y_pred_tree)
jaccard_score_tree = jaccard_score(y_test, y_pred_tree)
log_loss_tree = log_loss(y_test, y_pred_tree)


print("F1_score value in Decision tree model:", round(f1_score_tree, 4))
print("Jaccard score value in Decision tree model:", round(jaccard_score_tree, 4))
print("Log loss value in Decision tree model:", round(log_loss_tree, 4))


confusion_matrix_tree = plot_confusion_matrix(tree_model, x_test, y_test)


from sklearn import svm

svm = svm.SVC(C = 3, kernel = "rbf")
SVM_model = svm.fit(x_train, y_train)

y_pred_svm = SVM_model.predict(x_test)


f1_score_svm = f1_score(y_test, y_pred_svm)
jaccard_score_svm = jaccard_score(y_test, y_pred_svm)
log_loss_svm = log_loss(y_test, y_pred_svm)


print("F1_score value in Support Vector Machine model:", round(f1_score_svm, 4))
print("Jaccard score value in SVM model:", round(jaccard_score_svm, 4))
print("Log loss value in SVM model:", round(log_loss_svm, 4))


confusion_matrix_svm = plot_confusion_matrix(SVM_model, x_test, y_test)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 3, solver = "liblinear")

log_reg_model = log_reg.fit(x_train, y_train)

y_pred_logistic = log_reg_model.predict(x_test)


f1_score_logistic = f1_score(y_test, y_pred_logistic)
jaccard_score_logistic = jaccard_score(y_test, y_pred_logistic)
log_loss_logistic = log_loss(y_test, y_pred_logistic)


print("F1_score value in Logistic regression model:", round(f1_score_logistic, 4))
print("Jaccard score value in Logistic regression model:", round(jaccard_score_logistic, 4))
print("Log loss value in logistic regression model:", round(log_loss_logistic, 4))


confusion_matrix_logistic = plot_confusion_matrix(log_reg_model, x_test, y_test)


data_model= [["KNN", f1_score_KNN, jaccard_score_KNN, log_loss_KNN],
             ["Decision Tree", f1_score_tree, jaccard_score_tree, log_loss_tree],
             ["SVM", f1_score_svm, jaccard_score_svm, log_loss_svm],
             ["Logistic Regression", f1_score_logistic, jaccard_score_logistic, log_loss_logistic]]
df_data_model = pd.DataFrame(data_model, columns = ["Model", "F1 Score", "Jaccard Score", "Log Loss Score"])


df_data_model


sns.barplot(x = "Model", y = "F1 Score", data = df_date_model)


sns.barplot(x = "Model", y = "Jaccard Score", data = df_date_model)


sns.barplot(x = "Model", y = "Log Loss Score", data = df_date_model)
