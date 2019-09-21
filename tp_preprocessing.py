#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>
import numpy as np
import sys
from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_selection import chi2, f_classif

# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Preprocessing():

    def __init__(self, data, lookup):

        self.data = data
        self.lookup = lookup


    def _process_features(self):

        # credit card
        self.data = self.data[self.data['payment_type']==1]
        # tip amount should be non-negative
        self.data = self.data[self.data['tip_amount']>=0]
        # total amount should be larger than 2.5
        self.data = self.data[self.data['total_amount']>2.5]

        # create new features
        # convert lpep pickup datetime to standard datetime format
        self.data['pickup_datetime'] = self.data['lpep_pickup_datetime'].apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        # extract day of month
        self.data['monthday'] = self.data['pickup_datetime'].apply(lambda x: x.day)
        # get weekday
        self.data['weekday'] = self.data['pickup_datetime'].apply(lambda x: x.weekday() + 1)
        # get hour of day
        self.data['hour'] = self.data['pickup_datetime'].apply(lambda x: x.hour)
        # trip duration
        self.data['dropoff_datetime'] = self.data['lpep_dropoff_datetime'].apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        self.data['trip_duration'] = (self.data['dropoff_datetime'] - self.data['pickup_datetime']) / np.timedelta64(1, 'm')
        # remove duration less than 0
        self.data = self.data[self.data['trip_duration'] > 0]
        # derive average trip speed with miles per hour
        self.data['trip_speed'] = self.data['trip_distance'] / (self.data['trip_duration'] / 60.0)
        # remove speed greater than 80
        self.data = self.data[self.data['trip_speed'] <= 80]

        # remove 0 passenger
        self.data = self.data[self.data['passenger_count'] != 0]
        # merge 7/8/9 to 6 
        self.data['passenger_count'] = self.data['passenger_count'].apply(lambda x: 6 if x >= 6 else x)


    def _get_target(self):

        self.data['is_tipped'] = self.data['tip_amount'].apply(lambda x: 0 if x == 0 else 1)


    def _remove(self):

        # remove irrelevant features
        features = ['ehail_fee', 'VendorID', 'payment_type', 'lpep_pickup_datetime',
                    'lpep_dropoff_datetime', 'pickup_datetime', 'dropoff_datetime',
                    'store_and_fwd_flag', 'tip_amount', 'total_amount']
        self.data.drop(features, axis=1, inplace=True)


    def _discretize(self):

        # join pickup
        tmp = self.data.merge(self.lookup[['LocationID', 'Borough']], 
            left_on="PULocationID", right_on="LocationID", how="left")
        # join dropoff
        self.data = tmp.merge(self.lookup[['LocationID', 'Borough']], 
            left_on="DOLocationID", right_on="LocationID", how="left")
        # remove redundant features
        self.data.drop(columns=["PULocationID", "DOLocationID", "LocationID_x", "LocationID_y"], 
            axis=1, inplace=True)
        # rename location features
        self.data.rename(columns={"Borough_x":"PULocation", "Borough_y":"DOLocation"}, 
            inplace=True)
        # create a feature of whether pickup location is the same as drop off
        # 1: different, 0: same
        self.data["PU_DO"] = (self.data["PULocation"]!=self.data["DOLocation"]).astype(int)

        # label encoder pickup and dropoff location
        le = preprocessing.LabelEncoder()
        le.fit(self.data['PULocation'])
        self.data['PULocation'] = le.transform(self.data['PULocation'])
        self.data['DOLocation'] = le.transform(self.data['DOLocation'])


    def _feature_selection(self):

        features = [x for x in self.data.columns if x not in ['is_tipped']]
        X = self.data[features].values
        y = self.data['is_tipped'].values
        score, _ = chi2(X, y)
        score = np.nan_to_num(score)
        res = sorted(zip(*(features, score)), key=lambda x: x[1], reverse=True)
        # remove two features with the least score
        selected = [x[0] for x in res[:-2]] 
        self.data = self.data[selected+['is_tipped']]


    def main(self):

        print ("========== Start preprocessing ==========")
        # print ("Remove some samples...")
        self._process_features()
        self._get_target()
        self._remove()
        self._discretize()
        self._feature_selection()

        # check missing value
        if self.data.isnull().sum().sum() != 0:
            print ("There are still %d missing values in the dataset."%self.data.isnull().sum().sum())
            sys.exit()

        return self.data
