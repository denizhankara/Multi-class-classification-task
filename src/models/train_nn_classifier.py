import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
import matplotlib.pylab as pl
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,log_loss
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import cross_val_predict,StratifiedKFold,cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from BorutaShap import BorutaShap
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from collections import Counter
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns
# multiple models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.model_selection import GridSearchCV

from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing



def trainModel(model,X_train,y_train,trained_model_name):
    # trained model name is the model name for the trained model
    import pickle
    model.fit(X_train,y_train)

    file_name = trained_model_name+".pkl"

    pickle.dump(model, open(file_name, "wb"))

    return model

def makePredictions(model,X_test,y_test):
    # make predictions on trained model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    ll = log_loss(y_test, y_proba)
    print("Log_loss: %f" % ll)

    #Confusion matrix
    fig = plt.figure()
    cnf_matrix = confusion_matrix(y_test, predictions)
    disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, values_format='.0f')
    print(cnf_matrix)
    plt.savefig('../../src/visualization/confusion_matrix_one_vs_rest_with_class_weights_SVMSVC.png')
    plt.close(fig)
    return y_pred


# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def get_custom_model(n_inputs,n_outputs):
    model = Sequential()
    model.add(Dense(40, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(10,kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

    pass

def evaluate_model(X, y,x_unseen= None,y_unseen= None):
    results = list()
    results_logloss = list()
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    #cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=1,random_state=42)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        #X_train, X_test = X[train_ix], X[test_ix]
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        #model = get_model(n_inputs, n_outputs)
        model = get_custom_model(n_inputs,n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)


        if True:
            # Make prediction on unseen dataset
            lb = preprocessing.LabelBinarizer()
            y_unseen = lb.fit_transform(y_unseen)

            yhat = model.predict(x_unseen)
            y_proba = model.predict_proba(x_unseen)
            # round probabilities to class labels
            yhat = yhat.round()


            # calculate accuracy
            acc = accuracy_score(y_unseen, yhat)
            # store result
            print('accuracy > %.3f' % acc)
            results.append(acc)

            # calculate log loss
            ll = log_loss(y_unseen, y_proba)
            # store result
            print('logloss > %.3f' % ll)
            results_logloss.append(ll)

        if False:
            # make a prediction on the test set


            yhat = model.predict(X_test)
            y_proba=model.predict_proba(X_test)
            # round probabilities to class labels
            yhat = yhat.round()


            # calculate accuracy
            acc = accuracy_score(y_test, yhat)
            # store result
            print('accuracy > %.3f' % acc)
            results.append(acc)

            # calculate log loss
            ll = log_loss(y_test, y_proba)
            # store result
            print('logloss > %.3f' % ll)
            results_logloss.append(ll)

    return results,results_logloss


if __name__ == '__main__':
    train_path = "../features/corrAndPvalue.csv"
    # train_path = "../features/correlationEliminated.csv"
    # train_path = "../../data/raw/trainData.csv"
    test_path = "../../data/raw/testData.csv"

    train_data = pd.read_csv(train_path)
    # test_data = pd.read_csv(test_path)

    labels = train_data['target']
    # del train_data['target']
    class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    print(sorted(Counter(labels).items()))

    # Smote section
    smote_train_path = "../../data/raw/smoteData.csv"
    smote_train_data = pd.read_csv(smote_train_path)
    smote_labels = smote_train_data['target']
    print(sorted(Counter(smote_labels).items()))
    del smote_train_data['target']

    labels = train_data['target']

    del train_data['target']
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42,stratify=labels)


    # evaluate model
    results,results_logloss = evaluate_model(smote_train_data, smote_labels,X_test,y_test)
    # summarize performance
    print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
    print('Logloss: %.3f (%.3f)' % (mean(results_logloss), std(results_logloss)))