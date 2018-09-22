#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	K-means class

from copy import deepcopy
import numpy as np
import pandas as pd

from base_model import BaseModel

#=============================
# KMeans
#
# - Class to encapsulate a model for testing
#=============================
class KMeans(BaseModel) :

	#=============================
	# KMeans()
	#
	#	- constructor
	#@param	data	input data as pandas data frame
	#@param	num_clusters	number of clusters to train
	#=============================
	def __init__(self, data, num_clusters):
		BaseModel.__init__(self, data.values)
		self.num_clusters = num_clusters
		#initialize centroids to be random data points in data set (use numpy)
		self.centroids = data.sample(num_clusters).values 
		self.num_data_rows = len(self.data) # convenience
		#initialize clusters to effectively empty
		self.clusters = np.zeros(self.num_data_rows)

	#=============================
	# get_euclidean_distance_vector()
	#
	#	- get the distance between two matrixes
	#		- Note the axis will be x, y, or NONE 
	#			depending on what val you want to calculate
	#=============================
	def get_euclidean_distance_vector(self, input_matrix_1, input_matrix_2, axis_in):
		distance_vector = np.linalg.norm((input_matrix_1 - input_matrix_2), axis=axis_in)
		return distance_vector

	#=============================
	# train()
	#
	#	- train on the data set
	#=============================
	def train(self):
		#print('KMeans class training')

		#initialize distance between centroids to be not 0
		distance_between_new_and_old_centroids = 999999.0

		while distance_between_new_and_old_centroids != 0.0:
			#Find the clusters via closest distance
			for row in range(self.num_data_rows):
				#Get the distances from the clusters across all features
				x_axis = 1 #get the x_axis distances between features
				point_distances = \
					self.get_euclidean_distance_vector(self.data[row], self.centroids, x_axis)
				#Get the index of the min val of distances
				cluster_idx = np.argmin(point_distances)
				#Use the index as the cluster name! Brilliant!
				self.clusters[row] = cluster_idx

			#save centroids so you can compare if they moved!
			prev_centroids = deepcopy(self.centroids)

			#new centroids as mean value of their points
			for cluster_idx in range(self.num_clusters):
				#Get the data points corresponding to the current cluster
				cluster_data_points = \
						[self.data[row] for row in range(self.num_data_rows) \
							if self.clusters[row] == cluster_idx]
				y_axis = 0 #take the mean of the values across all rows
				cluster_mean = np.mean(cluster_data_points, axis=y_axis)
				self.centroids[cluster_idx] = cluster_mean

			#Get the total distance between two matrixes
			distance_between_new_and_old_centroids = \
					self.get_euclidean_distance_vector(self.centroids, prev_centroids, None)

		return self.centroids, self.clusters

	#=============================
	# evaluate()
	#
	#	- evaluate the model 
	#
	#@return	value of performance
	#=============================
	def evaluate(self):
		print('Evaluate Cluster Training')
		#TODO: Silhouette FCN
		return -1


#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main - test kmeans')
	input_data = pd.DataFrame([[1.0,1.1],[3.3,5.5],[1.2,0.9],[3.0,5.3]])
	print('Input data:')
	print(input_data)
	print('expected output is Centroids c1{1.1, 1.0} && c2{3.15, 5.4} & Clusters {1,3} && {2,4} ')
	num_clusters = 2
	kmeans_model = KMeans(input_data, num_clusters)
	centroids, clusters = kmeans_model.train()
	print('centroids')
	print(centroids)
	print('clusters')
	print(clusters)

	input_data = pd.DataFrame([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],[9,9,9,9]])
	print('Input data:')
	print(input_data)
	num_clusters = 3
	kmeans_model = KMeans(input_data, num_clusters)
	centroids, clusters = kmeans_model.train()
	print('centroids')
	data_frame_centroid = pd.DataFrame(centroids)
	print(centroids)
	print('clusters')
	print(clusters)

if __name__ == '__main__':
	main()
