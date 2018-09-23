#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Run machine learning algorithms

import argparse
from file_manager import FileManager
from stepwise_forward_selection import StepwiseForwardSelection

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to run tests')

	parser = argparse.ArgumentParser(description='Run Stepwise Forward Selection && K-Means clustering on given data')
	parser.add_argument('file_path', type=str, help='full path to input file')
	parser.add_argument('num_clusters', type=int, default=2, nargs='?', help='number of total clusters')
	args = parser.parse_args()

	#INPUTS
	print()
	print('INPUTS')
	file_path = args.file_path
	print('input file path:', file_path)
	num_clusters = args.num_clusters
	print('number of clusters: ', num_clusters)
	print()

	#READ INPUT DATA
	input_data = FileManager.get_csv_file_data_pandas(file_path)
	print('head of input data')
	print(input_data.head())
	print()

	#PREPROCESS DATA
	#According to 'names' files && visual inspection there are 0 missing values for these data sets
	#Remove missing value rows
	#input_data.pd.dropna(inplace=True)
	#print()

	#RUN FEATURE SELECTION && K-MEANS
	sfs = StepwiseForwardSelection(input_data)
	chosen_features, chosen_data_set = sfs.run_sfs(num_clusters)
	print()
	print('RESULT:')
	print('Chosen features (indexed from 0):')
	print(chosen_features)
	print('Chosen data:')
	print(chosen_data_set)
	print('Best silhouette coefficient:')
	print(sfs.base_performance)
	print('Clusters:')
	print(sfs.chosen_model.clusters)
	print('Centroids:')
	print(sfs.chosen_model.centroids)
	print()


if __name__ == '__main__':
	main()
