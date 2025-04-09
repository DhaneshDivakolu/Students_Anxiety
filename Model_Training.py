
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

final_depression_dataset = pd.read_csv("depression_dataset.csv")
final_stress_dataset = pd.read_csv("stress_dataset.csv")
final_anxiety_dataset = pd.read_csv("anxiety_dataset.csv")

#Seperate the data and labels
depression_labels = final_depression_dataset["Label"]
depression_X = final_depression_dataset.drop(columns=["Label"])

depression_labels

depression_X

#Encode the labels
encoder = LabelEncoder()
encoded_depression_label = encoder.fit_transform(depression_labels)

# Define the desired label values
desired_label_values = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']
encoder.classes_ = desired_label_values

dict(zip(encoder.classes_,range(len(encoder.classes_))))

encoded_depression_label

dp_X_Train, dp_X_Test, dp_Y_Train, dp_Y_Test = train_test_split(depression_X, encoded_depression_label, test_size=0.3, random_state= 30)

dp_X_Train

dp_Y_Train

dp_Y_Test

# Calculate the count of each unique label
unique_labels, label_counts = np.unique(dp_Y_Test, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Test data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")

# Calculate the count of each unique label
unique_labels, label_counts = np.unique(dp_Y_Train, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Training data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")

#Seperate the data and labels
stress_labels = final_stress_dataset["Label"]
stress_X = final_stress_dataset.drop(columns=["Label"])

stress_X

#Encode the labels
encoder = LabelEncoder()
encoded_stress_label = encoder.fit_transform(stress_labels)

# Define the desired label values
desired_label_values = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']
encoder.classes_ = desired_label_values

dict(zip(encoder.classes_,range(len(encoder.classes_))))

encoded_stress_label

#Get the training and test set from the stress dataset
st_X_Train, st_X_Test, st_Y_Train, st_Y_Test = train_test_split(stress_X, encoded_stress_label, test_size=0.3, random_state= 30)

st_X_Train

st_Y_Train

st_X_Test

st_Y_Test

# Calculate the count of each unique label
unique_labels, label_counts = np.unique(st_Y_Test, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Test data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")

# Calculate the count of each unique label
unique_labels, label_counts = np.unique(st_Y_Train, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Training data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")

#Seperate the data and labels
anxiety_labels = final_anxiety_dataset["Label"]
anxiety_X = final_anxiety_dataset.drop(columns=["Label"])

anxiety_X

#Encode the labels
encoder = LabelEncoder()
encoded_anxiety_label = encoder.fit_transform(anxiety_labels)

# Define the desired label values
desired_label_values = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']
encoder.classes_ = desired_label_values

dict(zip(encoder.classes_,range(len(encoder.classes_))))

encoded_anxiety_label

#Get the training and test set from the stress dataset
ax_X_Train, ax_X_Test, ax_Y_Train, ax_Y_Test = train_test_split(anxiety_X, encoded_anxiety_label, test_size=0.3, random_state= 30)

ax_X_Train

ax_Y_Train

ax_X_Test

ax_Y_Test

# Calculate the count of each unique label
unique_labels, label_counts = np.unique(ax_Y_Test, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Test data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")

# Calculate the count of each unique label
unique_labels, label_counts = np.unique(ax_Y_Train, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Training data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")

"""# **Model Training**

##KNN Model
"""

#Training the model on depression dataset
k_model = KNeighborsClassifier(n_neighbors=10)
k_model.fit(dp_X_Train, dp_Y_Train)

dp_predictions = k_model.predict(dp_X_Test)

dp_predictions

#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for depression using KNN")
plt.savefig('KNN Depression.png')
plt.show()

#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))

# Evaluate the model using accuracy, precision, recall, and F1-score
kd_accuracy = accuracy_score(dp_Y_Test, dp_predictions)
kd_precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
kd_recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
kd_f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(kd_accuracy*100))
print("Precision: %.f" %(kd_precision*100))
print("Recall: %.f" %(kd_recall*100))
print("F1-score: %.f" %(kd_f1*100))

"""**Model training on Stress dataset**"""

#Training the model on depression dataset
k_model = KNeighborsClassifier(n_neighbors=10)
k_model.fit(st_X_Train, st_Y_Train)

st_predictions = k_model.predict(st_X_Test)

st_predictions

#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for stress using KNN")
plt.savefig('KNN Stress.png')
plt.show()

#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
ks_accuracy = accuracy_score(st_Y_Test, st_predictions)
ks_precision = precision_score(st_Y_Test, st_predictions, average='macro')
ks_recall = recall_score(st_Y_Test, st_predictions, average='macro')
ks_f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(ks_accuracy*100))
print("Precision: %.f" %(ks_precision*100))
print("Recall: %.f" %(ks_recall*100))
print("F1-score: %.f" %(ks_f1*100))

#Training the model on Anxiety dataset
k_model = KNeighborsClassifier(n_neighbors=10)
k_model.fit(ax_X_Train, ax_Y_Train)

ax_predictions = k_model.predict(ax_X_Test)

ax_predictions

#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for Anxiety using KNN")
plt.savefig('KNN Anxiety.png')
plt.show()

#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
ka_accuracy = accuracy_score(ax_Y_Test, ax_predictions)
ka_precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
ka_recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
ka_f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(ka_accuracy*100))
print("Precision: %.f" %(ka_precision*100))
print("Recall: %.f" %(ka_recall*100))
print("F1-score: %.f" %(ka_f1*100))

"""# SVM Model

**Model training on Depression dataset**
"""

#Training the model on depression dataset
svm_model = SVC(C = 10, kernel = 'rbf', gamma= 0.2, random_state=24)
svm_model.fit(dp_X_Train, dp_Y_Train)

dp_predictions = svm_model.predict(dp_X_Test)

dp_predictions

#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for depression using svc")
plt.savefig('SVM Depression.png')
plt.show()

#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))

# Evaluate the model using accuracy, precision, recall, and F1-score
sd_accuracy = accuracy_score(dp_Y_Test, dp_predictions)
sd_precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
sd_recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
sd_f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(sd_accuracy*100))
print("Precision: %.f" %(sd_precision*100))
print("Recall: %.f" %(sd_recall*100))
print("F1-score: %.f" %(sd_f1*100))
"""**Model training on Stress dataset**"""

#Training the model on stress dataset
svm_model = SVC(C = 10, kernel = 'rbf', gamma= 0.2, random_state=24)
svm_model.fit(st_X_Train, st_Y_Train)

st_predictions = svm_model.predict(st_X_Test)

st_predictions

#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for stress using svc")
plt.savefig('SVM Stress.png')
plt.show()

#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
ss_accuracy = accuracy_score(st_Y_Test, st_predictions)
ss_precision = precision_score(st_Y_Test, st_predictions, average='macro')
ss_recall = recall_score(st_Y_Test, st_predictions, average='macro')
ss_f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(ss_accuracy*100))
print("Precision: %.f" %(ss_precision*100))
print("Recall: %.f" %(ss_recall*100))
print("F1-score: %.f" %(ss_f1*100))

"""**Model training on Anxiety dataset**"""

#Training the model on depression dataset
svm_model = SVC(C = 10, kernel = 'rbf', gamma= 0.2, random_state=24)
svm_model.fit(ax_X_Train, ax_Y_Train)

ax_predictions = svm_model.predict(ax_X_Test)

ax_predictions

#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for Anxiety using svc")
plt.savefig('SVM Anxiety.png')
plt.show()

#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
sa_accuracy = accuracy_score(ax_Y_Test, ax_predictions)
sa_precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
sa_recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
sa_f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(sa_accuracy*100))
print("Precision: %.f" %(sa_precision*100))
print("Recall: %.f" %(sa_recall*100))
print("F1-score: %.f" %(sa_f1*100))

"""# Random Forest Model

**Model training on Depression dataset**
"""

#Training the model on depression dataset
rf_model = RandomForestClassifier(random_state = 30)
rf_model.fit(dp_X_Train, dp_Y_Train)

dp_predictions = rf_model.predict(dp_X_Test)

dp_predictions

#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for depression using Random Forest")
plt.savefig('RF Depression.png')
plt.show()

#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
rd_accuracy = accuracy_score(dp_Y_Test, dp_predictions)
rd_precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
rd_recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
rd_f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(rd_accuracy*100))
print("Precision: %.f" %(rd_precision*100))
print("Recall: %.f" %(rd_recall*100))
print("F1-score: %.f" %(rd_f1*100))

"""**Model training on Stress dataset**"""

#Training the model on stress dataset
rf_model = RandomForestClassifier(random_state=24)
rf_model.fit(st_X_Train, st_Y_Train)

st_predictions = rf_model.predict(st_X_Test)

st_predictions

#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for stress using Random Forest")
plt.savefig('RF Stress.png')
plt.show()

#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
rs_accuracy = accuracy_score(st_Y_Test, st_predictions)
rs_precision = precision_score(st_Y_Test, st_predictions, average='macro')
rs_recall = recall_score(st_Y_Test, st_predictions, average='macro')
rs_f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(rs_accuracy*100))
print("Precision: %.f" %(rs_precision*100))
print("Recall: %.f" %(rs_recall*100))
print("F1-score: %.f" %(rs_f1*100))

"""**Model training on Anxiety dataset**"""

#Training the model on depression dataset
rf_model = RandomForestClassifier(random_state=24)
rf_model.fit(ax_X_Train, ax_Y_Train)

ax_predictions = rf_model.predict(ax_X_Test)

ax_predictions

#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for Anxiety using Random Forest")
plt.savefig('RF Anxiety.png')
plt.show()

#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
ra_accuracy = accuracy_score(ax_Y_Test, ax_predictions)
ra_precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
ra_recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
ra_f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(ra_accuracy*100))
print("Precision: %.f" %(ra_precision*100))
print("Recall: %.f" %(ra_recall*100))
print("F1-score: %.f" %(ra_f1*100))

"""# Decision Tree Model

**Model training on Depression dataset**
"""

#Training the model on depression dataset
dt_model = DecisionTreeClassifier(random_state = 30, max_depth=20)
dt_model.fit(dp_X_Train, dp_Y_Train)

dp_predictions = dt_model.predict(dp_X_Test)

dp_predictions

#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for depression using Decision Tree")
plt.savefig('DT Depression.png')
plt.show()

#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
dd_accuracy = accuracy_score(dp_Y_Test, dp_predictions)
dd_precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
dd_recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
dd_f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(dd_accuracy*100))
print("Precision: %.f" %(dd_precision*100))
print("Recall: %.f" %(dd_recall*100))
print("F1-score: %.f" %(dd_f1*100))

"""**Model training on Stress dataset**"""

#Training the model on stress dataset
dt_model = DecisionTreeClassifier(random_state = 30, max_depth=20)
dt_model.fit(st_X_Train, st_Y_Train)

st_predictions = dt_model.predict(st_X_Test)

st_predictions

#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for stress using Decision Tress")
plt.savefig('DT Stress.png')
plt.show()

#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
ds_accuracy = accuracy_score(st_Y_Test, st_predictions)
ds_precision = precision_score(st_Y_Test, st_predictions, average='macro')
ds_recall = recall_score(st_Y_Test, st_predictions, average='macro')
ds_f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(ds_accuracy*100))
print("Precision: %.f" %(ds_precision*100))
print("Recall: %.f" %(ds_recall*100))
print("F1-score: %.f" %(ds_f1*100))

"""**Model training on Anxiety dataset**"""

#Training the model on depression dataset
dt_model = DecisionTreeClassifier(random_state = 30, max_depth=20)
dt_model.fit(ax_X_Train, ax_Y_Train)

ax_predictions = dt_model.predict(ax_X_Test)

ax_predictions

#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for Anxiety using Decision Tress")
plt.savefig('DT Anxiety.png')
plt.show()

#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
da_accuracy = accuracy_score(ax_Y_Test, ax_predictions)
da_precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
da_recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
da_f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(da_accuracy*100))
print("Precision: %.f" %(da_precision*100))
print("Recall: %.f" %(da_recall*100))
print("F1-score: %.f" %(da_f1*100))

"""# Naive Bayes Model

**Model training on Depression dataset**
"""

#Training the model on depression dataset
nb_model = GaussianNB()
nb_model.fit(dp_X_Train, dp_Y_Train)

dp_predictions = nb_model.predict(dp_X_Test)

dp_predictions

#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for Depression using Navy-Bayes")
plt.savefig('GNB Depression.png')
plt.show()

#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
nd_accuracy = accuracy_score(dp_Y_Test, dp_predictions)
nd_precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
nd_recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
nd_f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(nd_accuracy*100))
print("Precision: %.f" %(nd_precision*100))
print("Recall: %.f" %(nd_recall*100))
print("F1-score: %.f" %(nd_f1*100))

"""**Model training on Stress dataset**"""

#Training the model on stress dataset
nb_model = GaussianNB()
nb_model.fit(st_X_Train, st_Y_Train)

st_predictions = nb_model.predict(st_X_Test)

st_predictions

#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)
print("Confusion Matrix for stress using Navy-Bayes")
plt.savefig('GNB Stress.png')
plt.show()

#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
# Evaluate the model using accuracy, precision, recall, and F1-score
ns_accuracy = accuracy_score(st_Y_Test, st_predictions)
ns_precision = precision_score(st_Y_Test, st_predictions, average='macro')
ns_recall = recall_score(st_Y_Test, st_predictions, average='macro')
ns_f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(ns_accuracy*100))
print("Precision: %.f" %(ns_precision*100))
print("Recall: %.f" %(ns_recall*100))
print("F1-score: %.f" %(ns_f1*100))

"""**Model training on Anxiety dataset**"""

#Training the model on depression dataset
nb_model = GaussianNB()
nb_model.fit(ax_X_Train, ax_Y_Train)

ax_predictions = nb_model.predict(ax_X_Test)

ax_predictions

#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('GNB Anxiety.png')
plt.show()
print("Confusion Matrix for Anxiety using Navy-Bayes")
#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))
na_accuracy = accuracy_score(ax_Y_Test, ax_predictions)
na_precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
na_recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
na_f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(na_accuracy*100))
print("Precision: %.f" %(na_precision*100))
print("Recall: %.f" %(na_recall*100))
print("F1-score: %.f" %(na_f1*100))

models = ['Naive Bayes', 'KNN', 'Random Forest', 'Decision Tree', 'SVM']
accuracy_depression = [nd_accuracy, kd_accuracy, rd_accuracy, dd_accuracy, sd_accuracy]
accuracy_stress = [ns_accuracy, ks_accuracy, rs_accuracy, ds_accuracy, ss_accuracy]  
accuracy_anxiety = [na_accuracy, ka_accuracy, ra_accuracy, da_accuracy, sa_accuracy]

precision_depression = [nd_precision, kd_precision, rd_precision, dd_precision, sd_precision]
precision_stress = [ns_precision, ks_precision, rs_precision, ds_precision, ss_precision]  
precision_anxiety = [na_precision, ka_precision, ra_precision, da_precision, sa_precision]

recall_depression = [nd_recall, kd_recall, rd_recall, dd_recall, sd_recall]
recall_stress = [ns_recall, ks_recall, rs_recall, ds_recall, ss_recall]  
recall_anxiety = [na_recall, ka_recall, ra_recall, da_recall, sa_recall]

f1_depression = [nd_f1, kd_f1, rd_f1, dd_f1, sd_f1]
f1_stress = [ns_f1, ks_f1, rs_f1, ds_f1, ss_f1]  
f1_anxiety = [na_f1, ka_f1, ra_f1, da_f1, sa_f1]


# Plotting settings
x = np.arange(len(models))
width = 0.15

def plot_metrics(title, accuracy, precision, recall, f1):
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased height here
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='orange')
    ax.bar(x - 0.5*width, precision, width, label='Precision', color='tomato')
    ax.bar(x + 0.5*width, recall, width, label='Recall', color='hotpink')
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='violet')

    # Annotate bars with values
    for idx, values in enumerate([accuracy, precision, recall, f1]):
        offset = (-1.5 + idx) * width
        for i, val in enumerate(values):
            label = f'{val:.2f}' if val <= 1 else f'{val:.0f}'
            ax.text(i + offset, val + 0.01, label, ha='center', fontsize=10)  # Labels slightly above bars

    ax.set_title(f'Comparison of Model Performance - {title}', fontsize=16)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Scores", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'CMP {title}.png')
    plt.show()

# Plotting all three

plot_metrics("Depression", accuracy_depression, precision_depression, recall_depression, f1_depression)
plot_metrics("Stress", accuracy_stress, precision_stress, recall_stress, f1_stress)
plot_metrics("Anxiety", accuracy_anxiety, precision_anxiety, recall_anxiety, f1_anxiety)

