'''
This file accepts output csv files from the ZED camera, then uses the class confidence and
3d bounding box information to create a relation between those detected objects. This is done by
using a box and class loss method defined by the DETR model. Then, the hungarian algorithm is run
on the resulting matrix to give a relation with the lowest cost. Such is returned

@file network_relation.py
@version 11/2/2023
@author Intelligent Systems Laboratory @ UNCFSU
'''

import os
import numpy as np
import pandas as pd
import time

from utils import *

class InputFileException(Exception):
    """ This class is run when there is an issue with the input files """

def get_dataframes(root, files):
    '''This function takes in a list of filenames, extracts the data from them, puts each in a dataframe,
    then expands each dataframe to have the same number of objects (each should have the same number of
    objects as the dataframe with the highest inital number of objects)

    Parameters:
        root (String): The root directory where all the below files are held
        files (list of Strings): A list containing all filenames

    Returns:
        dataframes (list of Panda Dataframes): This is the list of pandas dataframes made from the input list of filenames
        max_objects_detected (int): The maximum number of objects detected across all dataframes (and subsequently the size of all df's)
        '''
    # Iterate through each file in the files, create DataFrames, and append each to the DataFrame list
    dataframes = [get_data(root, file) for file in files]

    # Obtain the largest number of detected objects across al DataFrames
    max_objects_detected = max([len(df['Object']) for df in dataframes])

    # Parse through each of the dataframes, then change the shape to match that of the max number of detected objects
    for df in dataframes:
        while df.shape[0] != max_objects_detected:
            # Every time this is ran, we are adding a row (object detected) with None values (The number of None values is determined by
            # by the number of attributed for each object (calculated by len(df.loc[0]))
            df.loc[len(df.index)] = [None] * len(df.loc[0])
    # Return finished list of DataFrames and max objects detected
    return dataframes, max_objects_detected

def get_object_lists(df_1, df_2):
    # Create lists of objects from each DataFrame, excluding rows where 'Class Confidence' or 3D_Bounding_Box' is None
    df_1_objectList = [item for i in range(len(df_1)) if df_1['Class Confidence'][i] is not None and df_1['3D_Bounding_Box'][i] is not None
                       for item in [[df_1['Class Confidence'][i], df_1['3D_Bounding_Box'][i]]]]
    df_2_objectList = [item for j in range(len(df_2)) if df_2['Class Confidence'][j] is not None and df_2['3D_Bounding_Box'][j] is not None
                       for item in [[df_2['Class Confidence'][j], df_2['3D_Bounding_Box'][j]]]]
    # Return lists of objects
    return df_1_objectList, df_2_objectList

def calculate_losses(temp, df_1_objectList, df_2_objectList, df_1, df_2):
    # Compute the loss for each pair of objects and save it in the temporary DataFrame
    for key1, node1 in enumerate(df_1_objectList):
        for key2, node2 in enumerate(df_2_objectList):
            temp.at[df_2['Object'][key2], df_1['Object'][key1]] = get_loss_2box(node1[1], node1[0], node2[1], node2[0])
    # Return the temporary DataFrame
    return temp

def network_relation(root, files):
    start = time.time() # Start the timer to calculate time required to compute network relation

    # Get dataframes and max number of objects detected
    dataframes, max_objects_detected = get_dataframes(root, files)

    # Check that all files are CSV files
    for file in files:
        if not file.endswith('.csv'):
            raise InputFileException("Input files but all be of type .csv")
    
    count = 0  # Used as a counter to keep track of the current interation over the DataFrames
    # Each iteration, we compare the first and second files in the files list, 
    # then remove the first one so that the next two are compared. Therefore
    # we keep repeating this process until there is only one file left, and
    # since we cannot compare it to a second one, it is the last and we stop
    while len(dataframes) > 1:
        # Select the first two DataFrames
        df_1 = dataframes[0]
        df_2 = dataframes[1]
        
        # Create a temporary DataFrame filled with infinity values, to be used in the Hungarian Algorithm
        temp = pd.DataFrame(float('inf'), index=df_2['Object'].to_list(), columns=df_1['Object'].to_list())
        # Create lists of objects from each DataFrame, excluding rows where 'Class Confidence' or 3D_Bounding_Box' is None
        df_1_objectList, df_2_objectList = get_object_lists(df_1, df_2) 
        # Compute the loss for each pair of objects and save it in the temporary DataFrame
        temp = calculate_losses(temp, df_1_objectList, df_2_objectList, df_1, df_2)
        # Replace infinity values with a large finite number (as infinity cannot be subtracted)
        temp.replace(float('inf'), np.finfo(np.float64).max, inplace=True)
        # Run the Hungarian algorithm on the temporary DataFrame 
        cols, rows, _ = hungarian(temp.to_numpy())
        # Filter out invalid assignments (those involving the large finite number)
        valid_assignments = temp.to_numpy()[cols, rows] != np.finfo(np.float64).max
        cols = cols[valid_assignments]
        rows = rows[valid_assignments]
        # Create a DataFrame to store the correspondences found by the Hungarian algorithm
        correspondence_df = pd.DataFrame({files[count].split('.csv')[0]: cols, files[count+1].split('.csv')[0]: rows})
        correspondence_df.index = ['Person_' + str(i) for i in range(len(correspondence_df))]
        # Remove the first DataFrame and proceed to the next pair
        dataframes = dataframes[1:]
        count += 1
        print(correspondence_df)
        
    end = time.time() # Stop the timer
    print(f"Total time: {end-start}") # Print the total execution time
    print(correspondence_df)
    
    # Save the final correspondence DataFrame to a CSV file
    correspondence_df.to_csv(root + 'output_relation.csv')
    
    # Return the sum of valid assignments and the column indices of these assignments
    return [temp.to_numpy()[cols, rows].sum(), cols]

def main():
    # Input filepaths here, the relation will be displayed in the terminal
    experimentName = r'Exp1A/'
    root = r'/home/mrrobot/Documents/ISL-Projects-main/TurtlebotZED/data_collection/10-31-23/'
    # Make final touches to filepath and obtaining filenames before proceeding with calculations
    finalFilepath = root + experimentName
    files = [filename for filename in os.listdir(finalFilepath) if (filename.endswith('.csv') and not filename.startswith('output'))]
    # Calculate and print relations
    print(network_relation(finalFilepath, files))

if __name__ == '__main__':
    main()
