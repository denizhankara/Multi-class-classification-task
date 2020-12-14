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
from sklearn.preprocessing import MultiLabelBinarizer



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

