import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import load_data as ld
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc

def svm_model(X_train, Y_train, x_test, y_test):
    # --------------------------------------------- MODEL 1: SVM

    # USE RANDOMIZED-SEARCH FOR HYPER-PARAMETERS SEARCH (kernel, degree if poly, C)
    # model = SVC()
    # random_param = {
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'degree': [3, 4],
    #     'C': np.arange(0.1,2,0.5)
    # }
    # randomized_search = RandomizedSearchCV(model, random_param, cv=5, scoring="accuracy", verbose=1, n_jobs=-1, n_iter = 10)
    # randomized_search.fit(X_train, Y_train)
    # #PRINT THE BEST PARAMETERS FOUND
    # print("Best parameters found:", randomized_search.best_params_) #C: 1.6, degree: 3, kernel: rbf

    # PREPARE THE SVC MODEL
    # kernel = randomized_search.best_params_['kernel'] #hyperparameter, used to compare the results;
    # C = randomized_search.best_params_['C']
    # degree = randomized_search.best_params_['degree']

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

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of SVM model: \n", conf_matrix)

    # PLOT RESULTS:
    # --- PRECISION-RECALL CURVE
    # precision_d = dict()
    # recall_d = dict()
    # for i in range(len(unique_class)):
    #     precision_d[i], recall_d[i], _ = precision_recall_curve(y_test == unique_class[i], y_pred == unique_class[i])
    #     plt.plot(recall_d[i], precision_d[i], lw=2, label='class {}'.format(unique_class[i]))
    #
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend(loc="best")
    # plt.title("Precision-Recall Curve")
    # plt.show()

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1','2','3','4','5','6','7','8','9'], yticklabels=['0', '1','2','3','4','5','6','7','8','9'])
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix SVM')
    plt.savefig('matrix_confusion_svm.png')
    plt.show()

    # --- ROC CURVE:
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(len(unique_class)):
    #     fpr[i], tpr[i], _ = roc_curve(y_test == unique_class[i], y_pred == unique_class[i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     plt.plot(fpr[i], tpr[i], lw=2, label='class {} (AUC = {:.2f})'.format(unique_class[i], roc_auc[i]))
    #
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend(loc="best")
    # plt.title("ROC Curve")
    # plt.show()

    return accuracy
def perceptron_model(X_train, Y_train, x_test, y_test):
    # FIND BEST PARAMETERS WITH GRIDSEARCH:
    model = Perceptron()
    random_param = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.001, 0.0001, 0.00001],
        'max_iter': np.arange(100, 1000, 200),
        'eta0': np.arange(0.01,0.1,0.05)
    }
    randomized_search = RandomizedSearchCV(model, random_param, cv=5, scoring="accuracy", verbose=1, n_jobs=-1, n_iter=10)
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

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of Decision Tree model: \n", conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix Perceptron')
    plt.savefig('matrix_confusion_perceptron.png')
    plt.show()

    return accuracy
def decision_tree_model(X_train, Y_train, x_test, y_test):
    model = tree.DecisionTreeClassifier()

    #FIT THE MODEL
    model.fit(X_train, Y_train)

    #PREDICT
    y_pred = model.predict(x_test)

    #METRICS
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of Decision Tree model: \n", accuracy)

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("Precision of Decision Tree model: \n", precision)
    print("Recall of Decision Tree model: \n", recall)

    f1 = f1_score(y_test, y_pred, average=None)
    print("F1_score of Decision Tree model: \n", f1)


    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of Decision Tree model: \n", conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix DT')
    plt.savefig('matrix_confusion_dt.png')
    plt.show()

    return accuracy


# LOAD THE DATASET
X, Y = ld.load_data('dataset1.csv')
x_data = np.array(X)
y = np.array(Y)
accuracies = [] #accuracies[0] = first_model, accuracies[1] = second_model, ecc..
print("Shape of x:", x_data.shape)
print("Shape of y:", y.shape)

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
print("Run Perceptron...\n")
accuracy2 = perceptron_model(X_train, Y_train, x_test, y_test)
print("Run Decision Tree...\n")
accuracy3 = decision_tree_model(X_train, Y_train, x_test, y_test)








