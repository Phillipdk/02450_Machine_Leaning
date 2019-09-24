## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load file in
file_path = './BreastTissue.xls'
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

# Deleting outliers
for pos, val in enumerate(data): 
    if val[4]>50000:
        data = np.delete(data, pos, 0)

# Saving in a txt file
np.savetxt("ordnet_data.csv", data, delimiter=",")

#[(plt.figure(), plt.plot(row)) for row in data.T[0:9]]


plt.figure(figsize=(10,10),dpi=100)
for i in range(len(attributeNames)-1):
    for j in range(len(attributeNames)-1):
        plt.subplot(9,9, i*9 +j +1)
        plt.plot(data[:,i], data[:,j], '.')
        







