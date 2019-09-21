#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>
import numpy as np
from sklearn.metrics import accuracy_score

# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Testing():

	def __init__(self, X_test, y_test):

		self.X_test = X_test
		self.y_test = y_test


	def test(self, model):

		y_pred = model.predict(self.X_test)
		accuracy = accuracy_score(self.y_test, y_pred)

		return accuracy

