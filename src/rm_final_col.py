#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Script to remove the final column of csv

import csv
import argparse

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to run tests')

	parser = argparse.ArgumentParser(description='Remove the final column')
	parser.add_argument('file_path_in', type=str, help='full path to input file')
	parser.add_argument('file_path_out', type=str, help='full path to output file')
	args = parser.parse_args()

	#INPUTS
	print()
	print('INPUTS')
	input_path = args.file_path_in
	print('input file path:', input_path)
	output_path = args.file_path_out
	print('output file path:', output_path)

	#STRIP FINAL COLUMN
	with open(input_path, "r") as file_in:
		with open(output_path, "w") as file_out:
			writer = csv.writer(file_out)
			for row in csv.reader(file_in):
				writer.writerow(row[:-1])


if __name__ == '__main__':
	main()
