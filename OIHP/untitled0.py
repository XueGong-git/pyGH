#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:54:07 2024

@author: gongxue
"""

import pandas as pd

# Load the data from the text file
file_path = './data/MAPbI3_Cubic_CNXPb_atmlist_L5_f993.txt'
data = pd.read_csv(file_path, delimiter='\t', header=None)  # Adjust the delimiter as needed

# Check for duplicate rows
duplicates = data[data.duplicated()]

if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)
else:
    print("No duplicate rows found.")