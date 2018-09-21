#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	StepwiseForwardSelection class

import numpy as np
import pandas as pd
import argparse
import operator
import copy
from test_model import TestModel

#=============================
# StepwiseForwardSelection
#
# - Class to encapsulate a StepwiseForwardSelection method
#=============================
class StepwiseForwardSelection:

	#=============================
	# StepwiseForwardSelection()
	#
	#	- perform stepwise forward selection on the given model 
	#	- Assumes columns (and rows) are labeled by their idx i.e. 0,1,2...
	#
	#@param		data	input data as Pandas DataFrame (assumed to have rows/cols 0,1,2...)
	#@return	winning model (encapsulates data & results)
	#=============================
	def __init__(self, data):
		#TODO: potentially strip the class column out of hte data_set
		#TODO: normalize data? 
		self.data = data

	#=============================
	# run_sfs()
	#
	#	- perform stepwise forward selection on the given model 
	#	- Assumes columns (and rows) are labeled by their idx i.e. 0,1,2...
	#
	#@return	winning features as a list (i.e. [0, 2, 5])
	#=============================
	def run_sfs(self, clusters=2):
		chosen_features = list() #list of column numbers corresponding to chosen features
		chosen_data_set = pd.DataFrame()

		base_performance = -2
		current_performance = -1
		best_performance = -1
		best_features = -1
		best_model = TestModel(self.data, clusters)
		chosen_model = best_model

		while (1):
			num_chosen_features = len(chosen_features) #can be used as next column index to add to data set
			print('number of chosen features:', num_chosen_features)
			print('Iterate over all features')
			print('------------------------')
			for column in self.data:
				#Account for (skip) features already been chosen as best
				if column in chosen_features:
					print('Already selected this column', column, ' Skippin')
					continue

				#Select a feature (column)
				chosen_features.append(column)
				print('chosen_features:')
				print(chosen_features)
				#Get the feature vector (column)
				feature_vector = self.data[column]
				#print('feature vector')
				#print(feature_vector)

				#Get the data set of ALL chosen 
				#chosen_data_set = pd.concat([chosen_data_set, feature_vector], axis=1)
				chosen_data_set[num_chosen_features] = feature_vector
				#print('chosen_data_set after concat:')
				#print(chosen_data_set)
				model = TestModel(chosen_data_set, 2)
				model.train()
				current_performance = model.evaluate()

				#TODO: May need to copy these values!
				print('best perf', best_performance, ' vs. current_perf', current_performance)
				if current_performance > best_performance:
					best_performance = current_performance
					best_model = model
					best_feature = column
					best_data = pd.DataFrame(chosen_data_set)
					print('best performance now current perf', best_performance)
					print('best_feature column', best_feature)
					print(best_data)

				#Remove the data & chosen feature & get next feature/data
				column_to_drop = len(chosen_features) - 1

				chosen_data_set.drop(chosen_data_set.columns[column_to_drop], axis=1, inplace=True)
				#print('chosen_data_set after drop:')
				#print(chosen_data_set)
				chosen_features.pop()
			print('------------------------')

			#TODO: may need to copy these values manually
			print('best perf', best_performance, ' vs. base_performance', base_performance)
			if best_performance > base_performance:
				base_performance = best_performance
				chosen_features.append(best_feature)
				chosen_data_set = best_data
				chosen_model = best_model
				print('base performance now best perf', base_performance)
				print('chosen feature column', best_feature)
				print(chosen_data_set)
			else:
				break

		print('base performance', base_performance)
		print('chosen features') 
		print(chosen_features)
		print(chosen_data_set)

#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing the stepwise_forward_selection algorithm')

	print()
	print('TEST 1: dummy data')
	print('input data1:')
	test_data = pd.DataFrame([[-1, 1, 2], [0, 1, 2]])
	#test_data2 = pd.DataFrame([[0, 1, -1, 2], [0, 1, -1, 2],[0, 1, -1, 2]])
	#test_data3 = pd.DataFrame([[1.0, 1.1],[3.3,5.5],[1.2,0.9],[3.0,5.3]])
	#TODO: use the professors example with known final solution
	print(test_data)

	sfs = StepwiseForwardSelection(test_data)
	chosen_features = sfs.run_sfs()
	#print('Winning model:')
	#print(chosen_features)


if __name__ == '__main__':
	main()
