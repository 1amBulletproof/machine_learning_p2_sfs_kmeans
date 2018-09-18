#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	TestModel class

import numpy as np
import pandas as pd
import argparse
import operator
import copy
from base_model import BaseModel

#=============================
# TestModel
#
# - Class to encapsulate a model for testing
#=============================
class TestModel(BaseModel) :

	def __init__(self, data_set)
		BaseModel.__init__(self, data_set)

	#=============================
	# train()
	#
	#	- train on the data set
	#=============================
	def train(self):
		print('Test class training')

	#=============================
	# evaluate()
	#
	#	- evaluate the model 
	#
	#@param		test_data	optional, can provide other data set to use
	#@return				value of performance
	#=============================
	def evaluate(self, test_data=-1):
		print('Test Class training')
		if (test_data == -1):
			print('No input test_data provided, evaluating performance on my training data')

		print('evaluating the test model, for now returning 1')
		#TODO: return the sum of the features
		return 1;

#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing test model')

	print()
	print('TEST 1: dummy data')
	print('input data1:')
	#TODO: turn this into dataframe
	test_data = [[0, 0, 0], [1, 1, 1], [ -1, -1, -1], [2, 2, 2]] #Should Select columns 0 && 1
	print(test_data)

	test_model = TestModel(test_data)
	test_model.train()
	result = test_model.evaluate()
	print('Result: ')
	print(result)


if __name__ == '__main__':
	main()
