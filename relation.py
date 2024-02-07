import os
import numpy as np
import pandas as pd

from utils import *

root = '/home/mrrobot/Documents/ISL-Projects-main/TurtlebotZED/data_collection'


def main():
	files = ['data_exp_testFiles-1pos_-3-2.83-2+rot_0-0-0.csv',
		'data_exp_testFiles-1pos_3-2.83--5+rot_0-0.79-0.csv',
		'data_exp_testFiles-1pos_0-2.83-0+rot_0-0-0.csv']
	n=0
	
	correspondence = pd.DataFrame(np.random.randint(0,5,size=(1, 3)), columns = [file.split('.csv')[0] for file in files], index=['Person_0'])
	correspondence[:] = np.nan

	for first_file in files:
		if not first_file.endswith('.csv'):
				continue
		for second_file in files:
			if not second_file.endswith('.csv'):
				continue
			
			if first_file == second_file:
				continue
			
			df_1 = get_data(first_file)
			df_2 = get_data(second_file)
			print("new frame")

			if df_1.shape[0] != df_2.shape[0]:
				print('Error')
				if df_1.shape[0] > df_2.shape[0]:
					df_2.loc[len(df_2.index)] = [None, None] 
				else:
					df_1.loc[len(df_1.index)] = [None, None] 

			temp = pd.DataFrame(np.random.random_sample(size=(1, 1)), columns = df_1['Object'].to_list(), index=df_2['Object'].to_list())

			for i in range(0,len(df_1)):
				for j in range(0,len(df_2)):
					u = df_1['3D_Bounding_Box'][i]
					v = df_2['3D_Bounding_Box'][j]
					print(u)
					print("-----")
					print(v)
					dist = hausdorff(u, v)

					temp.at[df_2['Object'][j], df_1['Object'][i]] = dist[0]

			#print(temp)
			cols, rows, _ = hungarian(temp.to_numpy())
			#print(first_file.split('.csv')[0], cols)
			#print(second_file.split('.csv')[0], rows)
			#print()
			

			if n == 0:
				correspondence[first_file.split('.csv')[0]] = cols
				correspondence[second_file.split('.csv')[0]] = rows
			else: 
				correspondence = update_correspondence(cols, rows, correspondence, first_file.split('.csv')[0], second_file.split('.csv')[0])
			n+=1	

				
		files=files[1:]
	correspondence = correspondence.astype(int)
	print(correspondence)

		
main()
