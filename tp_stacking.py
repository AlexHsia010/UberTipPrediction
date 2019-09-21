#!/usr/bin/env python
#-*- coding: utf-8 -*-


from sklearn.model_selection import KFold
import scipy
import numpy as np
import pickle

# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Stacking():

    def __init__(self):

        return


    def _get_oof(self, clf, X_train, y_train, X_test):

        ntrain = X_train.shape[0]
        ntest = X_test.shape[0]
        kf = KFold(n_splits= 3, random_state=42)
        
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((3, ntest))
        i = 0
        for train_index, test_index in kf.split(X_train):
            x_tr = X_train[train_index]
            y_tr = y_train[train_index]
            x_te = X_train[test_index]
    
            clf.fit(x_tr, y_tr)
    
            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(X_test)
            
            i += 1
        
        oof_test[:] = scipy.stats.mode(oof_test_skf, axis=0)[0]
        
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def stacking(self, tune_results, X_train, y_train, X_test):

        stack_train = []
        stack_test = []

        print ("========== Start stacking models ==========")
        for clf in tune_results:
            tmp = clf(**tune_results[clf][1])
            oof_train, oof_test = self._get_oof(tmp, X_train, y_train, X_test)
            stack_train.append(oof_train)
            stack_test.append(oof_test)

        X_train_stacked = np.concatenate(stack_train, axis=1)
        X_test_stacked = np.concatenate(stack_test, axis=1)

        # get a meta classifier
        clf0 = list(tune_results.keys())[0]
        meta_clf = clf0(**tune_results[clf0][1])
        # fit meta classifier
        meta_clf.fit(X_train_stacked, y_train)
        # save model
        with open('./model/meta_clf.pkl', 'wb') as clf:
            # print (tune_results[res][0])
            pickle.dump(meta_clf, clf)

        return meta_clf, X_test_stacked




    





















