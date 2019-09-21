#!/usr/bin/env python
#-*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.model_selection import GridSearchCV

# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Modeling():

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train


    def _model_selection(self, models):

        """
        Choose 3 best models among several candidates 
        using cross validation
        """
        print ("========== Start model selection ==========")
        res = []
        for model in models:
            scores = cross_val_score(model(), self.X_train, self.y_train, 
                cv=3, scoring='accuracy')
            res.append((model, scores.mean()))

        # sorted by average accuracy
        top3 = sorted(res, key=lambda x: x[1], reverse=True)[:3]

        return [x[0] for x in top3]


    def _hyperparam_tuning(self, models, dist_dic):

        tune_results = {}

        for clf in models:
            params = dist_dic[clf]
            tune = GridSearchCV(clf(),params,scoring="accuracy",cv=3,verbose=True,n_jobs=-1)
            print ("========== Start tuning %s =========="%clf.__name__)
            tune.fit(self.X_train, self.y_train)
            # store results
            tune_results[clf] = (tune.best_estimator_, tune.best_params_, tune.best_score_)

        return tune_results


    def save_model(self, tune_results):

        for res in tune_results:
            path = './model/'+res.__name__+'.pkl'
            # print (path)
            with open(path, 'wb') as clf:

                # print (tune_results[res][0])
                pickle.dump(tune_results[res][0], clf)





















