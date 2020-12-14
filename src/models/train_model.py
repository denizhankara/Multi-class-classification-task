import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
import matplotlib.pylab as pl
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
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
from sklearn.utils import class_weight


def makePredictions(model, X_test, y_test, class_names=None):
    # make predictions on trained model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    ll = log_loss(y_test, y_proba)
    print("Log_loss: %f" % ll)

    # Confusion matrix
    fig = plt.figure()
    cnf_matrix = confusion_matrix(y_test, predictions)
    disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, values_format='.0f')
    print(cnf_matrix)
    plt.savefig('../../src/visualization/confusion_matrix.png')
    plt.close(fig)
    return y_pred


def overSample(train_data, labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(sorted(Counter(labels).items()))

    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(random_state=42)
    train_data, labels = oversample.fit_resample(train_data, labels)
    print(sorted(Counter(labels).items()))
    return train_data, labels


def featureSelection(train_data, labels):
    # do correlation based feature seelction and such
    pass


def underSample(train_data, labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(sorted(Counter(labels).items()))

    from imblearn.under_sampling import RandomUnderSampler
    undersampe = RandomUnderSampler(random_state=42)
    train_data, labels = undersampe.fit_resample(train_data, labels)
    print(sorted(Counter(labels).items()))
    return train_data, labels


def smoteSample(train_data, labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(sorted(Counter(labels).items()))

    from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
    sm = SMOTE(random_state=42)
    train_data, labels = sm.fit_resample(train_data, labels)

    unique, counts = np.unique(labels, return_counts=True)
    print(sorted(Counter(labels).items()))

    saved_df = train_data.copy()
    saved_df['target'] = labels
    saved_df.to_csv("../../data/raw/smoteData.csv", index=False)

    return train_data, labels


def trainModel(model, X_train, y_train, trained_model_name):
    # trained model name is the model name for the trained model
    import pickle
    model.fit(X_train, y_train)

    file_name = trained_model_name + ".pkl"

    pickle.dump(model, open(file_name, "wb"))

    return model


def report_missing_values(train_data, labels):
    percent_missing = train_data.isnull().sum() * 100 / len(train_data)
    missing_value_df = pd.DataFrame({'column_name': train_data.columns,
                                     'percent_missing': percent_missing})

    missing_value_df.sort_values('percent_missing', inplace=True)
    print(missing_value_df)
    return missing_value_df


def correlationBasedFeatureSelection(train_data, labels, model=None):
    corr = train_data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                print(i, j)
                if columns[j]:
                    columns[j] = False
    selected_columns = train_data.columns[columns]
    train_data = train_data[selected_columns]
    selected_columns = selected_columns[:-1]
    # save the data after correlation elimination
    train_data.to_csv("../features/correlationEliminated.csv", index=False)

    # p-value column selection

    del train_data['target']
    import statsmodels.api as sm
    def backwardElimination(x, Y, sl, columns):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)

        regressor_OLS.summary()
        return x, columns

    SL = 0.05
    data_modeled, selected_columns = backwardElimination(train_data.values, labels.values, SL,
                                                         selected_columns)

    # result = pd.DataFrame()
    # result['target'] = labels

    data = pd.DataFrame(data=data_modeled, columns=selected_columns)
    data['target'] = labels
    #  save data to csv
    data.to_csv("../features/corrAndPvalue.csv", index=False)


def visualizetSNE(features, class_names):
    """this function does tsne visualization"""

    # labels = features['target']
    # del features['target']
    # np.random.seed(42)
    # rndperm = np.random.permutation(features.shape[0])
    N = 10000
    df_subset = features.sample(N, random_state=42)
    labels = df_subset['target']
    del df_subset['target']
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(df_subset)
    print('Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca_50.explained_variance_ratio_)))
    tsne = TSNE(n_components=2, verbose=0, perplexity=320, n_iter=900)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    #  make the t-sne plot
    target_ids = range(1, len(class_names) + 1)

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange'
    for i, c, label in zip(target_ids, colors, class_names):
        plt.scatter(tsne_pca_results[labels == i, 0], tsne_pca_results[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.savefig('../../src/visualization/tsne_representation_uncorrelated_data.png')
    plt.close(fig)

    # plt.show()

    # fig = plt.figure()
    # plt.savefig('../../src/visualization/raw_class_distributions.png')
    # plt.close(fig)

    pass


def tuneXGBModel(model, train_data, labels):
    """
    init_model = xgb.XGBClassifier(learning_rate=0.5,
                                   n_estimators=150,
                                   max_depth=6,
                                   min_child_weight=0,
                                   gamma=0,
                                   reg_lambda=1,
                                   subsample=1,
                                   colsample_bytree=0.75,
                                   scale_pos_weight=1,
                                   objective='multi:softmax',  # multi-softprob
                                   num_class=9,
                                   random_state=42)
    """

    param_test = {
        'classification__max_depth': range(4, 9, 2),
        'classification__min_child_weight': range(0, 5, 2),
        'classification__colsample_bytree': [i / 10.0 for i in range(6, 8)],
        'classification__reg_lambda': [0.1, 0.5, 1],
        'classification__learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7],
        'classification__n_estimators': [100, 200, 300, 400]

    }

    train_data = train_data.drop(["target"], axis=1, errors='ignore')

    from imblearn.pipeline import Pipeline
    model = Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('classification', model)
    ])

    gsearch = GridSearchCV(estimator=model,
                           param_grid=param_test, scoring='neg_log_loss', n_jobs=5, iid=False, cv=5)
    gsearch.fit(train_data, labels)

    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)
    # file_name = "tuned_xgb_model.pkl"
    # pickle.dump(model, open(file_name, "wb"))
    return gsearch.best_params_


def extractTrainAndTestDataForLabel(train_data, test_data):
    #  extract the classed labels

    confused_classes_train = train_data.loc[
        (train_data['target'] == 2) | (train_data['target'] == 3) | (train_data['target'] == 4)]
    confused_classes_test = test_data.loc[
        (test_data['target'] == 2) | (test_data['target'] == 3) | (test_data['target'] == 4)]

    return confused_classes_train, confused_classes_test


def transformDatasetAndTrain(X_train, y_train, X_test, y_test, class_names):
    train_data = pd.concat([X_train, y_train, ], axis=1)
    train_data.loc[train_data.target != 2, "target"] = 0

    y = train_data['target']
    del train_data['target']
    # y_test.loc[y_test==0]
    model = xgb.XGBClassifier(learning_rate=0.3,
                              n_estimators=150,
                              max_depth=6,
                              min_child_weight=0,
                              gamma=0,
                              reg_lambda=1,
                              subsample=1,
                              colsample_bytree=0.75,
                              scale_pos_weight=1,
                              objective='multi:softmax',  # multi-softprob
                              num_class=2,
                              random_state=42,

                              # class_weight ={2:1,3:2}
                              )
    model = trainModel(model, train_data, y, "oneVsRest")
    makePredictions(model, X_test, y_test, ["0", "2"])

    return model


def sklearnoneVsRestClassifier(X_train, X_test, y_train, y_test):
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    from sklearn.svm import LinearSVC
    from sklearn import svm

    confused_classifier = xgb.XGBClassifier(learning_rate=0.3,
                                            n_estimators=150,
                                            max_depth=6,
                                            min_child_weight=0,
                                            gamma=0,
                                            reg_lambda=1,
                                            subsample=1,
                                            colsample_bytree=0.75,
                                            scale_pos_weight=1,
                                            objective='multi:softmax',  # multi-softprob
                                            num_class=9,
                                            random_state=42,

                                            # class_weight ={2:1,3:2}
                                            )
    full_classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                                  random_state=0))
    y_score = full_classifier.fit(X_train, y_train).score(X_test, y_test)
    print(y_score)

    # Get all accuracies
    classes = np.unique(y_train)

    def get_acc_single(clf, X_test, y_test, class_):
        pos = np.where(y_test == class_)[0]
        neg = np.where(y_test != class_)[0]
        y_trans = np.empty(X_test.shape[0], dtype=bool)
        y_trans[pos] = True
        y_trans[neg] = False
        return clf.score(X_test, y_trans)  # assumption: acc = default-scorer

    for class_index, est in enumerate(full_classifier.estimators_):
        class_ = classes[class_index]
        print('class ' + str(class_))
        print(get_acc_single(est, X_test, y_test, class_))

    pass
    # start making predictions
    # make predictions on trained model

    y_pred = full_classifier.predict(X_test)
    y_proba = full_classifier.predict_proba(X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Confusion matrix
    class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    fig = plt.figure()
    cnf_matrix = confusion_matrix(y_test, predictions)
    disp = plot_confusion_matrix(full_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, values_format='.0f')
    print(cnf_matrix)
    plt.savefig('../../src/visualization/confusion_matrix_one_vs_rest_with_class_weights_SVMSVC.png')
    plt.close(fig)

    ll = log_loss(y_test, y_proba)
    print("Log_loss: %f" % ll)

    # makePredictions(full_classifier, X_test, y_test)
    pass


def trainOnevsRestClassifier(X_train, X_test, y_train, y_test, class_label):
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    class_names = ["2", "3", "4"]
    conf_train, conf_test = extractTrainAndTestDataForLabel(train_data, test_data)

    #  form a smaller classifier

    confused_classifier = xgb.XGBClassifier(learning_rate=0.3,
                                            n_estimators=150,
                                            max_depth=6,
                                            min_child_weight=0,
                                            gamma=0,
                                            reg_lambda=1,
                                            subsample=1,
                                            colsample_bytree=0.75,
                                            scale_pos_weight=1,
                                            objective='multi:softmax',  # multi-softprob
                                            num_class=9,
                                            random_state=42,

                                            # class_weight ={2:1,3:2}
                                            )

    #  See the data
    visualizetSNE(conf_train, class_names=class_names)

    conf_train_labels = conf_train['target']
    del conf_train['target']

    conf_test_labels = conf_test['target']
    del conf_test['target']

    model = trainModel(confused_classifier, conf_train, conf_train_labels, "confused_model_small")
    makePredictions(model, conf_test, conf_test_labels, class_names=class_names)

    return


def convertDataset(train_data, classes):
    # convert dataset to have only given classes and zero
    # train_data = train_data.loc[train_data['target'] not in classes, "target"] = 0
    train_data['target'] = train_data['target'].apply(lambda x: x if (x in classes) else 0)

    return train_data


def trainModelAndMakePredictions(init_model, train_data, labels):
    weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    init_model = xgb.XGBClassifier(learning_rate=0.3,
                                   n_estimators=150,
                                   max_depth=6,
                                   min_child_weight=0,
                                   gamma=0,
                                   reg_lambda=1,
                                   subsample=1,
                                   colsample_bytree=0.75,
                                   scale_pos_weight=1,
                                   objective='multi:softmax',  # multi-softprob
                                   num_class=9,
                                   random_state=42,
                                   class_weight=weights
                                   )

    # find classes in target column
    classes = sorted(set(list(labels)))
    class_names = list(map(str, classes))  #  convert to str for names
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)

    model = trainModel(init_model, X_train, y_train, "regular")
    makePredictions(model, X_test, y_test, class_names=class_names)

    model_y_pred_train = model.predict_proba(train_data)
    return model, model_y_pred_train


def makeSubmission(final_model,train_data,train_labels):

    test_data = pd.read_csv("../../data/raw/testData.csv")
    model =trainModel(final_model,train_data,labels,"intermediate-model-submitted")

    y_pred= model.predict_proba(test_data)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv("../../reports/intermediate_test_results.csv",index=False)
    return y_pred



# Resolve multi-class label ordering problem

if __name__ == "__main__":
    # train_path = "../features/corrAndPvalue.csv"
    #train_path = "../features/correlationEliminated.csv"
    train_path = "../../data/raw/trainData.csv"
    test_path = "../../data/raw/testData.csv"

    train_data = pd.read_csv(train_path)
    # test_data = pd.read_csv(test_path)

    classes = [i for i in range(1, 10)]

    if True:
        converted_classes = [2]
        converted_train_data = convertDataset(train_data.copy(), converted_classes)
        converted_labels = converted_train_data['target']
        del converted_train_data['target']

    else:
        converted_classes = classes
        converted_train_data = train_data
        converted_labels = train_data['target']
        del converted_train_data['target']

    labels = train_data['target']
    del train_data['target']

    # del train_data['target']
    class_names = list(map(str, classes))  #  convert to str for names
    converted_class_names = list(map(str, converted_classes))

    print(sorted(Counter(labels).items()))
    print(sorted(Counter(converted_labels).items()))

    # matplotlib.use('Agg')





    """
    ### VISUALIZATIONS

    # check class distributions
    unique, counts = np.unique(labels, return_counts=True)
    fig = plt.figure()
    plt.bar(unique, counts, 1)
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    #plt.draw()
    #plt.show()
    plt.savefig('../../src/visualization/raw_class_distributions.png')
    plt.close(fig)
    """

    # TSNE plot
    # visualizetSNE(train_data,class_names)
    # exit()

    # class imbalance-not neccesary

    """
    unique, counts = np.unique(labels, return_counts=True)
    print(sorted(Counter(labels).items()))

    #nm1 = NearMiss(version=1)
    #train_data, labels = nm1.fit_resample(train_data, labels)
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(random_state=42)
    train_data,labels = oversample.fit_resample(train_data,labels)
    print(sorted(Counter(labels).items()))
    """

    # Feature importance and selection
    """
    rfe = RFECV(estimator=xgb.XGBClassifier())
    model =xgb.XGBClassifier()
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    n_scores = cross_val_score(pipeline, train_data, labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    """

    # Correlation check
    """
    cor = train_data.corr()
    #plt.figure(figsize=(10,6))
    #sns.heatmap(cor,annot=True)
    cor_target = abs(cor["target"])
    relevant_features = cor_target[cor_target > 0.1]
    keys =list(relevant_features.keys())
    new_data = train_data[keys]
    train_data=new_data
    del train_data['target']
    
    """

    # Trying out multiple models
    """

    dfs = []
    models = [
          ('LogReg', LogisticRegression()),
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()),
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())
        ]
    results = []
    names = []
    scoring = ['accuracy']
    target_names = class_names# ['malignant', 'benign']
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, train_data, labels, cv=kfold, scoring=scoring)
        print(name)
        print(cv_results)
        #clf = model.fit(train_data, labels)
        #y_pred = clf.predict(X_test)
        #print(name)
        #print(classification_report(y_test, y_pred, target_names=target_names))
        #results.append(cv_results)
        #names.append(name)
        #this_df = pd.DataFrame(cv_results)
        #this_df['model'] = name
        #dfs.append(this_df)
    #final = pd.concat(dfs, ignore_index=True)

    """

    ## Start training

    # Borutashap feature selection
    """
    # if classification is False it is a Regression problem
    model = xgbc
    Feature_Selector = BorutaShap(model=model,
                                  importance_measure='shap',
                                  classification=True)

    Feature_Selector.fit(X=train_data, y=labels, n_trials=100, sample=False,
                         train_or_test='test', normalize=True,
                         verbose=True)

    # Returns Boxplot of features
    Feature_Selector.plot(which_features='all')

    # Returns a subset of the original data with the selected features
    subset = Feature_Selector.Subset()

    """

    # Tune RF model
    """

    #model = RandomForestClassifier()

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestClassifier(random_state=42)
    from pprint import pprint

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(rf.get_params())

    from sklearn.model_selection import RandomizedSearchCV

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train_data, labels)

    best_params = {'n_estimators': 1000,
     'min_samples_split': 2,
     'min_samples_leaf': 1,
     'max_features': 'auto',
     'max_depth': 50,
     'bootstrap': False}

    best_estimator_params = {'bootstrap': False,
     'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': 50,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 1000,
     'n_jobs': None,
     'oob_score': False,
     'random_state': 42,
     'verbose': 0,
     'warm_start': False}
    #print(rf_random.best_params_)
    
    
    
    best_estimator_params = {'bootstrap': False,
                             'ccp_alpha': 0.0,
                             'class_weight': None,
                             'criterion': 'gini',
                             'max_depth': 50,
                             'max_features': 'auto',
                             'max_leaf_nodes': None,
                             'max_samples': None,
                             'min_impurity_decrease': 0.0,
                             'min_impurity_split': None,
                             'min_samples_leaf': 1,
                             'min_samples_split': 2,
                             'min_weight_fraction_leaf': 0.0,
                             'n_estimators': 1000,
                             'n_jobs': None,
                             'oob_score': False,
                             'random_state': 42,
                             'verbose': 0,
                             'warm_start': False}
    model = RandomForestClassifier()
    model.set_params(**best_estimator_params)

    """
    #  Tune xgb model
    weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    init_model = xgb.XGBClassifier(learning_rate=0.3,
                                   n_estimators=150,
                                   max_depth=6,
                                   min_child_weight=0,
                                   gamma=0,
                                   reg_lambda=1,
                                   subsample=1,
                                   colsample_bytree=0.75,
                                   scale_pos_weight=1,
                                   objective='multi:softmax',  # multi-softprob
                                   num_class=9,
                                   random_state=42,
                                   # class_weight ={1:8,2:1,3:2,4:8,5:8,6:1.5,7:8,8:2,9:3}
                                   )

    # returned = tuneXGBModel(init_model,train_data,labels)
    # print(returned)
    """
    Accuracy: 82.16 %
    Log_loss: 0.518926
    Accuracy: 93.10 %
    Log_loss: 0.215931
    """

    # exit()
    # train_data,labels = correlationBasedFeatureSelection(train_data,labels,init_model)
    """Make a submission"""

    makeSubmission(init_model,train_data,labels)
    exit()

    """use a calibrated classifier"""
    from sklearn.calibration import CalibratedClassifierCV

    # trainModelAndMakePredictions(init_model, train_data, labels)
    calibrated = CalibratedClassifierCV(init_model, method='sigmoid', cv=3)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    scores = cross_val_score(init_model, train_data, labels, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    # summarize performance
    print(scores)
    print('Mean ROC AUC: %.3f' % np.mean(scores))

    exit()

    """ Tune by hand """
    # smote_train_path = "../../data/raw/smoteData.csv"
    # train_data = pd.read_csv(train_path)
    # test_data = pd.read_csv(test_path)
    # smote_train_data = pd.read_csv(smote_train_path)
    # smote_labels = smote_train_data['target']

    # print(sorted(Counter(smote_labels).items()))

    # del smote_train_data['target']

    # X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42,
    #                                                    stratify=labels)

    #  training phase
    # model = trainOnevsRestClassifier(X_train,X_test,y_train,y_test,class_label=[2,3,4])
    # model= trainModel(init_model,c)

    # model = transformDatasetAndTrain(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, class_names=[2])
    # sklearnoneVsRestClassifier(X_train,X_test,y_train,y_test)
    model_1, model_1_y_pred = trainModelAndMakePredictions(init_model, train_data, labels)

    model_2, model_2_y_pred = trainModelAndMakePredictions(init_model, converted_train_data, converted_labels)

    # new_train_data = train_data.copy()
    # new_train_data['model_1']=model_1_y_pred
    # new_train_data['model_2'] = model_2_y_pred
    new_train_data = train_data.join(pd.DataFrame(model_1_y_pred), rsuffix="model_1")
    new_train_data = new_train_data.join(pd.DataFrame(model_2_y_pred), rsuffix="model_2")

    model_3, model_3_y_pred = trainModelAndMakePredictions(init_model, new_train_data, labels)

    exit()

    model = trainModel(init_model, X_train, y_train, "regular")
    makePredictions(model, X_test, y_test)
    # exit()

    """
    X_train_sampled,y_train_sampled = underSample(X_train,y_train)
    model=trainModel(init_model,X_train_sampled,y_train_sampled,"undersampled")

    X_train_sampled, y_train_sampled = overSample(X_train, y_train)
    model = trainModel(init_model, X_train_sampled, y_train_sampled, "oversampled")
    """

    # X_train_sampled, y_train_sampled = smoteSample(X_train, y_train)
    # model = trainModel(init_model, smote_train_data, smote_labels, "smotesampled")

    """
    print("Undersampled model:")
    model = pickle.load(open("undersampled.pkl","rb"))
    makePredictions(model,X_test,y_test)
    makePredictions(model,train_data,labels)

    print("Oversampled model:")
    model = pickle.load(open("oversampled.pkl", "rb"))
    makePredictions(model, X_test, y_test)
    makePredictions(model, train_data, labels)
    """
    print("Smotesampled model:")
    # model = pickle.load(open("smotesampled.pkl", "rb"))
    # makePredictions(model, X_test, y_test)
    # makePredictions(model, train_data, labels)

    # Tune XGBoost Model

    # model = xgbc
    # model= pickle.load(open("xgb_oversampled.pkl","rb"))

    ## Start training

    # model = model.fit(X_train, y_train, eval_metric='mlogloss')
    # mcl = model.fit(X_train, y_train)

    # makePredictions(model,X_test,y_test)
    # makePredictions(model,train_data,labels)
    #  confusion matrix
    exit()
    fig = plt.figure()

    cnf_matrix = confusion_matrix(y_test, predictions)
    disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, values_format='.5f')
    print(cnf_matrix)
    plt.savefig('../../src/visualization/confusion_matrix_one_vs_rest_with_class_weights_SVMSVC.png')
    plt.close(fig)
    # plt.show()
    #  cross -validation
    cv = StratifiedKFold(n_splits=5, random_state=42)
    scores = cross_val_score(model, train_data, labels, cv=cv, scoring='neg_log_loss')
    print(scores)
    # proba = cross_val_predict(xgbc, train_data, labels, cv=cv, method='predict_proba')

    pass
