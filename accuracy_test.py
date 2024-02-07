import time
import math
import pandas as pd
import numpy as np
from utils import *

root = '/home/mrrobot/Documents/ISL-Projects-main/TurtlebotZED/data_collection'
csvfile = "data_exp_testFiles-1pos_-3-2.83-2+rot_0-0-0.csv"

## Here is where we define where each person should be. Note that the order in the list matters, and should be according to the order in the csv file
person0 = Node(np.array([-1, (69/12)/2, -10])) #Eric
#person2 = Node(np.array([-7, (69/12)/2, -10])) Toby
#person1 = Node(np.array([-3, (66/12)/2, -10])) Ashley

peopleTruth = [person0]

def convert(list1):   ## converting nasty str to a usable int format
	if list1 == None:
		return list1
	listFinal = []
	listTemp = []
	neg = [list1[1],list1[13],list1[25]]
	listTemp.extend([list1[2:12],list1[14:24],list1[26:36]])
	for x in range(3):
		if neg[x] == '-':
			listFinal.append(-float(listTemp[x]))
		else:
			listFinal.append(float(listTemp[x]))
	return listFinal


def positionalAccuracy(personTruthCoords, personExpCoords):
	## returns the 3d coordinate distance between the personTruthCoords and personExpCoords
	return np.linalg.norm(personTruthCoords.getPosition() - convert(personExpCoords))


def main(fileName):

	## place csv data in pandas DataFrame
	df = get_data(fileName)
	
	## Summate the total differences between each detected person's coordinates and where they actually were
	totalPosDiff = sum([positionalAccuracy(peopleTruth[x], df["Object_Position"][x]) for x in range(len(peopleTruth))])
	
	## Find the average difference per person in the experiment
	personPosDiff = totalPosDiff / len(peopleTruth)
	
	print(personPosDiff)
	

main(os.path.join(root, csvfile))
'''
https://pyimagesearch.com/2023/05/22/detr-breakdown-part-
1-introduction-to-detection-transformers/

https://pyimagesearch.com/2023/06/12/detr-breakdown-part-
2-methodologies-and-algorithms/

https://pyimagesearch.com/2023/06/26/detr-breakdown-part-
3-architecture-and-details/
'''
