#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	TestModel class

import numpy as np
import pandas as pd
import argparse
import operator
import random
import copy
from base_model import BaseModel

#=============================
# TestModel
#
# - Class to encapsulate a model for testing
#=============================
class TestModel(BaseModel) :

	def __init__(self, data, num_clusters):
		BaseModel.__init__(self, data)
		self.num_clusters = num_clusters

	#=============================
	# train()
	#
	#	- train on the data set
	#=============================
	def train(self):
		#print('Test class training')
		self.clusters = np.array_split(self.data, self.num_clusters)
		return


	#=============================
	# evaluate()
	#
	#	- evaluate the model 
	#
	#@return				value of performance
	#=============================
	def evaluate(self):
		#print('Evaluate Cluster Training')
		sum = 0
		for cluster in self.clusters:
			sum += cluster.values.sum()
			
		return sum


#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing test model')

	print()
	print('TEST 1: dummy data')
	print('input data1:')
	#TODO: turn this into dataframe
	test_data = pd.DataFrame([[0, 1, -1], [0, 1, -1],[0, 1, -1]])
	print(test_data)
	print()

	test_model = TestModel(test_data)
	test_model.train()
	result = test_model.evaluate()
	print('Result (vector of sums): ')
	print(result)


if __name__ == '__main__':
	main()
