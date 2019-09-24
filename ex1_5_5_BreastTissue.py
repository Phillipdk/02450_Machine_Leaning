## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load file in
file_path = '../Data/BreastTissue.xls'
breast_data = pd.read_excel(file_path, sheet_name = "Data", index_col=0)

# Show the attributes
attributeNames = np.array(breast_data.columns)

# Isolate the class types
tissue_names = np.array(breast_data.Class)
breast_data = breast_data.drop(['Class'], axis=1)

# Lav 1-out-of-K coding
tiss = []
[tiss.append(elem) for elem in tissue_names if elem not in tiss]

K = np.zeros((len(breast_data), len(tiss)))

for i in range(len(tissue_names)):
    K[i][tiss.index(tissue_names[i])] = 1

# Slå 1-out-of-K sammen data
data = np.concatenate((breast_data, K), axis=1)
del K, i, tiss, tissue_names, file_path


[(plt.figure(), plt.plot(row)) for row in data.T[0:9]]

wildgefesgfhrsfgrgkg
