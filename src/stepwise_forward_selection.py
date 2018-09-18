#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	StepwiseForwardSelection class

import numpy as np
import pandas as pd
import argparse
import operator
import copy

#=============================
# StepwiseForwardSelection
#
# - Class to encapsulate a StepwiseForwardSelection method
#=============================
class StepwiseForwardSelection:

	def __init__(self, model, data_set)
		self.model = model
		self.data = data_set

	#=============================
	# run_sfs()
	#
	#	- perform stepwise forward selection on the given model 
	#
	#@return	winning feature vector (i.e. [0, 2, 5])
	#=============================
	def run_sfs(self):
		print('start')

		#TODO:split up data by column (i.e. feature)

#=============================
# MAIN PROGRAM
#=============================
def main():
	pring('Main() - testing the stepwise_forward_selection algorithm')

	print()
	print('TEST 1: dummy data')
	print('input data1:')
	#TODO: turn this into dataframe
	test_data = [[0, 0, 0], [1, 1, 1], [ -1, -1, -1], [2, 2, 2]] #Should Select columns 0 && 1
	print(test_data)

	sfs = StepwiseForwardSelection(test_model, test_data)
	chosen_features = sfs.run_sfs()
	print('Chosen features:')
	print(chosen_features)


if __name__ == '__main__':
	main()
