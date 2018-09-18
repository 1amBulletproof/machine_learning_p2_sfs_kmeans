#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	BaseModel class

import numpy as np
import pandas as pd
import argparse
import operator
import copy

#=============================
# BaseModel
#
# - Base class of a model / classifier 
#=============================
class StepwiseForwardSelection:

	def __init__(self, data_set)
		self.data = data_set

	#=============================
	# train()
	#
	#	- train on the data set
	#=============================
	def train(self):
		print('Base Class training')

	#=============================
	# evaluate()
	#
	#	- evaluate the model 
	#
	#@param		test_data	optional, can provide other data set to use
	#@return				value of performance
	#=============================
	def evaluate(self, test_data=-1):
		print('Base Class training')
		if (test_data == -1):
			print('No input test_data provided, evaluating performance on my training data')

		print('evaluating the base model, no such thing!')
		return -1;
	

#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing base model - no such thing')


if __name__ == '__main__':
	main()
