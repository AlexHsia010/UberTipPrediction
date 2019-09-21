#!/usr/bin/env python

import numpy as np
import pandas as pd
from tp_preprocessing import Preprocessing
from tp_modeling import Modeling
from tp_stacking import Stacking
from tp_test import Testing

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# skip all warnings
import warnings
warnings.filterwarnings('ignore')


models = [LogisticRegression, RandomForestClassifier, AdaBoostClassifier, \
		  XGBClassifier, KNeighborsClassifier, ExtraTreesClassifier]
dist_dic = {
	
	LogisticRegression: {'C': np.linspace(0.1,1,5,endpoint=True),
						 'penalty': ['l2']},

	RandomForestClassifier: {'n_estimators': [300,350],
							 'max_depth': [10,None],
							 'min_samples_split': np.arange(2,5),
							 'min_samples_leaf':[1,2]
							 },

	AdaBoostClassifier: {'n_estimators':[300,350,400], 
						 'learning_rate':[0.1,0.4,0.7]
						 },

	XGBClassifier: {'n_estimators':[300,350],
    				'learning_rate':np.arange(0.1,1,0.3),
    				'max_depth':[3],
    				'min_child_weight':[2],
    				'gamma':[0.2],
    				'subsample':[0.8]},

    KNeighborsClassifier: {'n_neighbors':[5,7],
    					   'leaf_size':[2,4],
    					   'algorithm':['auto','kd_tree']},

    ExtraTreesClassifier: {'n_estimators':[300,350],
						   'max_depth': [10,None],
						   'min_samples_split': np.arange(2,5),
						   'min_samples_leaf':[1,2]}

}


if __name__ == '__main__':

    ######## Read and preprocess data ########
    trips = pd.read_csv('./data/green_tripdata_2018-12.csv')
    lookup = pd.read_csv('./data/taxi+_zone_lookup.csv')
    preprocess = Preprocessing(trips, lookup)
    data = preprocess.main()
    # train test split
    y = data['is_tipped'].values
    X = data.drop(['is_tipped'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

    ######## Hyperparameter tuning ########
    ml = Modeling(X_train, y_train)
    selected_models = ml._model_selection(models)
    tune_results = ml._hyperparam_tuning(selected_models, dist_dic)

    ######## save individual models ########
    ml.save_model(tune_results)

    ######## stacking ########
    stk = Stacking()
    meta_clf, X_test_stacked = stk.stacking(tune_results, X_train, y_train, X_test)

    ######## testing ########
    # test for individual models
    test = Testing(X_test, y_test)
    for model in tune_results:
    	clf = tune_results[model][0]
    	print ("Accuracy for %s is %.4f"%(model.__name__, test.test(clf)))

    # test for stacking model
    test = Testing(X_test_stacked, y_test)
    print ("Accuracy for Stacking is %.4f"%test.test(meta_clf))









