import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import load_data as ld
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc

def svm_model(X_train, Y_train, x_test, y_test):
    # --------------------------------------------- MODEL 1: SVM

    # USE GRID-SEARCH FOR HYPER-PARAMETERS SEARCH (kernel, degree if poly, C)
    # model = SVC()
    # grid_param = {
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'degree': [3, 4],
    #     'C': np.arange(0.1,2,0.5)
    # }
    # grid_search = GridSearchCV(model, grid_param, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
    # grid_search.fit(X_train, Y_train)
    # #PRINT THE BEST PARAMETERS FOUND
    # print("Best parameters found:", grid_search.best_params_) #C: 1.6, degree: 3, kernel: rbf

    # PREPARE THE SVC MODEL
    # kernel = grid_search.best_params_['kernel'] #hyperparameter, used to compare the results;
    # C = grid_search.best_params_['C']
    # degree = grid_search.best_params_['degree']

    model = SVC(kernel='rbf', C=1.6)  # FOR THE REPORT, SHOW ALSO THE RESULTS WITH DIFFERENT HYPER-P.

    # FIT THE MODEL
    model.fit(X_train, Y_train)

    # PREDICT
    y_pred = model.predict(x_test)
    unique_class = np.unique(y_pred)
    print("Label predicted:\n", y_pred)

    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of SVM model:\n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of SVM model: \n", precision)
    print("Recall of SVM model: \n", recall)

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of SVM model: \n", f1)

    # PLOT RESULTS:
    # --- PRECISION-RECALL CURVE
    precision_d = dict()
    recall_d = dict()
    for i in range(len(unique_class)):
        precision_d[i], recall_d[i], _ = precision_recall_curve(y_test == unique_class[i], y_pred == unique_class[i])
        plt.plot(recall_d[i], precision_d[i], lw=2, label='class {}'.format(unique_class[i]))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall Curve")
    plt.show()

    # --- ROC CURVE:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(unique_class)):
        fpr[i], tpr[i], _ = roc_curve(y_test == unique_class[i], y_pred == unique_class[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {} (AUC = {:.2f})'.format(unique_class[i], roc_auc[i]))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("ROC Curve")
    plt.show()

    return accuracy
def perceptron_model(X_train, Y_train, x_test, y_test):
    # FIND BEST PARAMETERS WITH GRIDSEARCH:
    # model = Perceptron()
    # grid_param = {
    #     'n_iter_no_change': np.arange(40, 100, 10),
    #     'eta0': np.arange(0.01,0.1,0.05)
    # }
    # grid_search = GridSearchCV(model, grid_param, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
    # grid_search.fit(X_train, Y_train)
    # #PRINT THE BEST PARAMETERS FOUND
    # print("Best parameters found:", grid_search.best_params_) #eta0 = 0.01, n_iter = 50
    # SET PARAMETERS
    print(X_train.shape)
    n_iter = 50
    learning_rate = 0.01

    # PREPARE THE MODEL
    perceptron_model = Perceptron(n_iter_no_change=n_iter, eta0=learning_rate, random_state=0)

    # FIT THE MODEL
    perceptron_model.fit(X_train, Y_train)

    # PREDICTION
    y_pred = perceptron_model.predict(x_test)

    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of perceptron model: \n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of Perceptron model: \n", precision)
    print("Recall of Perceptron model: \n", recall)

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of Perceptron model: \n", f1)

    return accuracy
def least_squares_model(X_train, Y_train, x_test, y_test):
    model = SGDClassifier(loss="squared_error", alpha=0.01, max_iter=10000, random_state=0)

    #FIT THE MODEL
    model.fit(X_train, Y_train)

    #PREDICT
    y_pred = model.predict(x_test)

    #METRICS
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of Least-Squares model: \n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of Least-Squares model: \n", precision)
    print("Recall of Least-Squares model: \n", recall)

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of Least-Squares model: \n", f1)

    return accuracy


# LOAD THE DATASET
X, Y = ld.load_data('dataset1.csv')
x_data = np.array(X)
y = np.array(Y)
accuracies = [] #accuracies[0] = first_model, accuracies[1] = second_model, ecc..
print("Shape of x:", x_data.shape)
print("Shape of y:", y)

pre_p = input("Do you want to pre-process X? Type y/n.\n")
while pre_p != "y" and pre_p != "Y" and pre_p != "n" and pre_p != "N":
    print("This input is not valid!")
    pre_p = input("Do you want to pre-process X? Type y/n.\n")
if pre_p == "y" or pre_p == "Y":
    # PRE-PROCESSING X
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    print("Data pre-processed: \n")
    print(x_data)
    pre_p = True
elif pre_p == "n" or pre_p == "N":
    print("Data NOT pre-processed: \n")
    print(x_data)
    pre_p = False

#SPLIT THE DATA
X_train, x_test, Y_train, y_test = train_test_split(x_data, y, test_size= 0.2, random_state=1)

#TEST THE MODELS:
print("Run SVM...\n")
accuracy1 = svm_model(X_train, Y_train, x_test, y_test)
accuracies.append(accuracy1) # add the accuracy in the array, for the SVM model.
print("Run Perceptron...\n")
accuracy2 = perceptron_model(X_train, Y_train, x_test, y_test)
accuracies.append(accuracy2)
print("Run LS...\n")
accuracy3 = least_squares_model(X_train, Y_train, x_test, y_test)
accuracies.append(accuracy3)






