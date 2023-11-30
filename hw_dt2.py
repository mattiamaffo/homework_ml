import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

import load_data as ld


def svm_model(X_train, Y_train, x_test, y_test):

    # --------------------------------------------- MODEL 1: SVM

    # Use Randomized Search for hyperparameter search

    model = SVC(kernel= 'sigmoid', C=1.6, gamma= 'scale', class_weight= 'balanced')
    #param_randomized = {
    #    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #    'gamma': [1, 0.1, 0.01],
    #    'C': [0.1, 1, 10],
    #    'class_weight': ['balanced']
    #}

    # # Create a subset of D
    # subset_size = 100
    # x_data_sub, y_sub = shuffle(X_train, Y_train, random_state=42)[:subset_size]
    #
    # # Split the subset
    # X_train_sub, _, Y_train_sub, _ = train_test_split(x_data_sub, y_sub, test_size=0.2, random_state=1)


    #Hyper-parameter tuning
    #randomized_search = RandomizedSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1,
    #                                      n_iter=3)
    #randomized_search.fit(X_train, Y_train)

    #Best parameter
    #print("Best parameters found:", randomized_search.best_params_)

    #Create the model
    #best_params = randomized_search.best_params_
    #model = SVC(**best_params)

    # Fit of the model
    # Use of Bagging classifier
    bagging_classifier = BaggingClassifier(model, n_estimators=5, max_samples=0.2, random_state=42, n_jobs=-1)

    # Fit
    bagging_classifier.fit(X_train, Y_train)

    # Predict
    y_pred = bagging_classifier.predict(x_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of SVM model:\n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of SVM model (sigmoid, C = 1.6, degree=3) with bagging: \n", precision)
    print("Precision mean of SVM model (sigmoid, C = 1.6, degree=3) with bagging: \n", round(precision.mean(), 5))
    print("Recall of SVM model (sigmoid, C = 1.6, degree=3) with bagging: \n", recall)
    print("Recall mean of SVM model (sigmoid, C = 1.6, degree=3) with bagging: \n", round(recall.mean(), 5))

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of SVM model (sigmoid, C = 1.6, degree=3) with bagging: \n", f1)
    print("F1_score mean of SVM model (sigmoid, C = 1.6, degree=3) with bagging: \n", round(f1.mean(), 5))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of SVM model with Bagging:", conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1','2','3','4','5','6','7','8','9'], yticklabels=['0', '1','2','3','4','5','6','7','8','9'])
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix SVM')
    plt.savefig('matrix_confusion_svm_dt2.png')
    plt.show()

def perceptron_model(X_train, Y_train, x_test, y_test):
    # FIND BEST PARAMETERS WITH RANDOMIZED-SEARCH:

    model = Perceptron()
    random_param = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.001, 0.0001, 0.00001],
        'max_iter': np.arange(100, 1000, 200),
        'eta0': np.arange(0.01,0.1,0.05)
    }
    randomized_search = RandomizedSearchCV(model, random_param, cv=5, scoring="accuracy", verbose=1, n_jobs=-1, n_iter=3)
    randomized_search.fit(X_train, Y_train)

    #PRINT THE BEST PARAMETERS FOUND
    print("Best parameters found:", randomized_search.best_params_) #eta0 = 0.01, n_iter = 700, penalty = l1, alpha = 1e-05
    # SET PARAMETERS
    print(X_train.shape)
    n_iter = randomized_search.best_params_['max_iter']
    learning_rate = randomized_search.best_params_['eta0']
    penalty = randomized_search.best_params_['penalty']
    alpha = randomized_search.best_params_['alpha']

    # PREPARE THE MODEL
    perceptron_model = Perceptron(penalty=penalty, alpha=alpha, max_iter=n_iter, eta0=learning_rate, random_state=0)

    # # FIT THE MODEL
    # perceptron_model.fit(X_train, Y_train)
    #
    # # PREDICTION
    # y_pred = perceptron_model.predict(x_test)

    bagging_classifier = BaggingClassifier(perceptron_model, n_estimators=10, random_state=42, n_jobs=-1)

    # Fit
    bagging_classifier.fit(X_train, Y_train)

    # Predict
    y_pred = bagging_classifier.predict(x_test)

    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of perceptron model: \n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of Perceptron model: \n", precision)
    print("Precision mean of Perceptron model: \n", round(precision.mean(), 5))
    print("Recall of Perceptron model: \n", recall)
    print("Recall mean of Perceptron model: \n", round(recall.mean(), 5))

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of Perceptron model: \n", f1)
    print("F1_score mean of Perceptron model: \n", round(f1.mean(), 5))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of Decision Tree model: \n", conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix Perceptron')
    plt.savefig('matrix_confusion_perceptron_dt2.png')
    plt.show()
def forest_classifier_model(X_train, Y_train, x_test, y_test):

    forest_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Fit
    forest_classifier.fit(X_train, Y_train)

    # Predict
    y_pred = forest_classifier.predict(x_test)

    #METRICS
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of Forest Classifier model: \n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of Forest Classifier model: \n", precision)
    print("Precision mean of Forest Classifier: \n", round(precision.mean(), 5))
    print("Recall of Forest Classifier model: \n", recall)
    print("Recall mean of Forest Classifier: \n", round(recall.mean(), 5))

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of Forest Classifier model: \n", f1)
    print("F1_score mean of Forest Classifier: \n", round(f1.mean(), 5))


    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of Forest Classifier model: \n", conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix DT')
    plt.savefig('matrix_confusion_dt_dt2.png')
    plt.show()


# LOAD THE DATASET
X, Y = ld.load_data('dataset2.csv')
x_data = np.array(X)
y = np.array(Y)
print("Shape of x:", x_data.shape)
print("Shape of y:", y.shape)

pre_p = input("Do you want to pre-process X with Standard Scaler or Normalize? Type s/n.\n")
while pre_p != "s" and pre_p != "S" and pre_p != "n" and pre_p != "N":
    print("This input is not valid!")
    pre_p = input("Do you want to pre-process X with Standard Scaler or Normalize? Type s/n.\n")
if pre_p == "s" or pre_p == "S":
    # PRE-PROCESSING X
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    print("Data pre-processed with Robust Scaler: \n")
    print(x_data)
    pre_p = True
elif pre_p == "n" or pre_p == "N":
    x_data = normalize(x_data, norm='l2')
    print("Data pre-processed with Normalize: \n")
    print(x_data)
    pre_p = False

#SPLIT THE DATA
X_train, x_test, Y_train, y_test = train_test_split(x_data, y, test_size= 0.2, random_state=1)

#TEST THE MODELS:
print("Run SVM...\n")
svm_model(X_train, Y_train, x_test, y_test)
print("Run Perceptron...\n")
perceptron_model(X_train, Y_train, x_test, y_test)
print("Run Forest Classifier...\n")
forest_classifier_model(X_train, Y_train, x_test, y_test)






