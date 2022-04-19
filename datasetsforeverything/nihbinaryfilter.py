# import pandas, shutil, os
import pandas as pd
import shutil
import os
import random
import math
import glob
# allow multiple output per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

print("Multiple outputs per cell")

# variables to determine the source and destination paths
# NEEDS TO HAVE THE ALL IMAGES
TRAIN_DIRECTORY = 'C:/Users/akash/Desktop/Akash/MSOEML/CheXpert/NIH'

TEST_DIRECTORY = 'C:/Users/akash/Desktop/Akash/MSOEML/CheXpert/NIH_Test'

if not os.path.exists(TEST_DIRECTORY):
    os.makedirs(TEST_DIRECTORY)
if not os.path.exists(TRAIN_DIRECTORY):
    os.makedirs(TRAIN_DIRECTORY)
if not os.path.exists(os.path.join(TEST_DIRECTORY, 'Abnormal')):
    os.makedirs(os.path.join(TEST_DIRECTORY, 'Abnormal'))
if not os.path.exists(os.path.join(TEST_DIRECTORY, 'Normal')):
    os.makedirs(os.path.join(TEST_DIRECTORY, 'Normal'))
if not os.path.exists(os.path.join(TRAIN_DIRECTORY, 'Abnormal')):
    os.makedirs(os.path.join(TRAIN_DIRECTORY, 'Abnormal'))
if not os.path.exists(os.path.join(TRAIN_DIRECTORY, 'Normal')):
    os.makedirs(os.path.join(TRAIN_DIRECTORY, 'Normal'))
# read in data and inspect it
images = pd.read_csv('NIH.csv')
# make the Image Index column as the index
images.set_index('Image Index', inplace=True)
images.head()
for fileName in os.listdir(TRAIN_DIRECTORY):
    if fileName in ['Abnormal', 'Normal']:
        continue
    if fileName not in images.index:
        continue

    # move the file to the appropriate directory
    sp = os.path.join(TRAIN_DIRECTORY, fileName)
    dp = ""
    if images.loc[fileName]['Finding Labels'] == 'No Finding':
        dp = os.path.join(TEST_DIRECTORY, 'Normal')
    else:
        dp = os.path.join(TRAIN_DIRECTORY, 'Abnormal')
    _ = shutil.move(sp, dp)
