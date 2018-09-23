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
		BaseModel.__init__(self, data.values )
		self.num_data_rows = len(self.data) # convenience
		self.num_clusters = num_clusters
		self.clusters = np.zeros(self.num_data_rows) # init clusters to empty

		# init centroids to be random data points in data set, but ensure all values are unique (otherwise empty clusters)
		self.centroids = data.sample(num_clusters).values 
		unique_centroids = np.unique(self.centroids, axis=0)
		loop_counter = 0
		self.valid_input = True # In case you want to sort into more clusters than you have unique pts
		while (unique_centroids.size != self.centroids.size):
			self.centroids = data.sample(num_clusters).values
			unique_centroids = np.unique(self.centroids, axis=0)
			loop_counter += 1
			if loop_counter > 1000:
				#print('WARNING: probably never finding a unique combination of starting centroids')
				self.valid_input = False
				break

	#=============================
	# get_euclidean_distance_vector()
	#
	#	- get the distance between two matrixes
	#		- Note the axis will be 0 (column-wise), 1(row-wise)x, None (computer 1 value) 
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
		if (self.valid_input == False):
			return self.centroids, self.clusters

		#initialize distance between centroids to be not 0
		distance_between_new_and_old_centroids = 999999.0

		while distance_between_new_and_old_centroids != 0.0:
			#Find the clusters via closest distance
			for row in range(self.num_data_rows):
				#Get the distances from the clusters across all features
				between_rows = 1 
				point_distances = \
					self.get_euclidean_distance_vector(self.data[row], self.centroids, between_rows)
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
			total_distance = None
			distance_between_new_and_old_centroids = \
					self.get_euclidean_distance_vector(self.centroids, prev_centroids, total_distance)

		return self.centroids, self.clusters

	#=============================
	# get_point_silhouette_coefficient()
	#=============================
	def get_point_silhouette_coefficient(self, row_idx):
		min_avg_distance_to_other_cluster = 1000 #init to arbitrary large value so all calculated values are less-than
		for cluster_idx in range(self.num_clusters):
			#Get the data points corresponding to the current cluster
			cluster_data_points = \
					[self.data[row] for row in range(self.num_data_rows) \
					if self.clusters[row] == cluster_idx]
			between_rows = 1
			distance_to_this_cluster = \
					self.get_euclidean_distance_vector(self.data[row_idx], cluster_data_points, between_rows)
			#IF this is my cluster, calculate the average distance to all points from me
			if cluster_idx == self.clusters[row_idx]:
				avg_distance_to_my_cluster = np.mean(distance_to_this_cluster)
			#IF this is NOT my cluster, calculate the average distance to all points from me	
			else:
				avg_distance_to_this_other_cluster = np.mean(distance_to_this_cluster)
				if avg_distance_to_this_other_cluster < min_avg_distance_to_other_cluster:
					min_avg_distance_to_other_cluster = avg_distance_to_this_other_cluster

		point_silhouette_coefficient = (min_avg_distance_to_other_cluster - avg_distance_to_my_cluster) / \
				max(avg_distance_to_my_cluster, min_avg_distance_to_other_cluster)

		return point_silhouette_coefficient

	#=============================
	# evaluate()
	#
	#	- evaluate the model 
	#
	#@return	value of performance
	#=============================
	def evaluate(self):
		if (self.valid_input == False):
			#print('WARNING: could not find unique starting centroid values')
			worst_performance_possible = -1
			return worst_performance_possible

		#Alternative methods
		#vector_get_point_silhouette_coefficient = np.vectorize(self.get_point_silhouette_coefficient)
		#point_silhouette_values = vector_get_point_silhouette_coefficient(self.data)

		point_silhouette_values = \
				np.array([self.get_point_silhouette_coefficient(point) for point in range(self.num_data_rows)])

		final_silhouette_value = np.mean(point_silhouette_values)

		return final_silhouette_value


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('Main - test kmeans')
	print()
	print('TEST: (2 clusters, small data)')
	input_data = pd.DataFrame([[1.0,1.1,4.0],[3.3,5.5,4.2],[1.2,0.9,4.1],[3.0,5.3, 4.1]])
	print('input data:')
	print(input_data)
	num_clusters = 2
	print()
	print('TRAIN: (expect cluster [A, B, A, B] && centroids [1.1,1] [3.15, 5.4])')
	kmeans_model = KMeans(input_data, num_clusters)
	centroids, clusters = kmeans_model.train()
	print('clusters:')
	print(clusters)
	print('centroids:')
	print(centroids)
	print()
	print('EVALUATE: silhouette value (-1 poor to +1 excellent clustering)')
	silhouette_value = kmeans_model.evaluate()
	print(silhouette_value)

	print()
	print()
	print('TEST: (3 clusters, more data)')
	input_data = pd.DataFrame([[1,1,1,1],[2,2,2,2],[3,3,3,3],[7,7,7,7],[8,8,8,8],[9,9,9,9],[13,13,13,13],[14,14,14,14],[15,15,15,15]])
	print('input data:')
	print(input_data)
	num_clusters = 3
	print()
	print('TRAIN: (expect cluster [A,A,A,B,B,B,C,C,C] && centroids [2-,], [8-], [14-]')
	kmeans_model = KMeans(input_data, num_clusters)
	centroids, clusters = kmeans_model.train()
	print('clusters:')
	print(clusters)
	print('centroids:')
	print(centroids)
	print()
	print('EVALUATE: silhouette value (-1 poor to +1 excellent clustering)')
	silhouette_value = kmeans_model.evaluate()
	print(silhouette_value)

if __name__ == '__main__':
	main()
