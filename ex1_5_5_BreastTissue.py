## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load file in
file_path = './BreastTissue.xls'
breast_data = pd.read_excel(file_path, sheet_name = "Data", index_col=0)

#del K, i, tiss, tissue_names, file_path

# Show the attributes 
attributeNames = np.array(breast_data.columns)

# Isolate the class types
tissue_names = np.array(breast_data.Class)
breast_data = breast_data.drop(['Class'], axis=1)

# Lav 1-out-of-K coding
tissue_types = []
[tissue_types.append(elem) for elem in tissue_names if elem not in tissue_types]

K = np.zeros((len(breast_data), len(tissue_types)))

for i in range(len(tissue_names)):
    K[i][tissue_types.index(tissue_names[i])] = 1

# Slå 1-out-of-K sammen data
data = np.concatenate((breast_data, K), axis=1)

# Remove outliers 
for pos, val in enumerate(data):
    if val[4]>50000:
        data = np.delete(data, pos, 0)

# Save to .txt file
np.savetxt("ordnet_data.csv", data, delimiter=",")

#[(plt.figure(), plt.plot(row)) for row in data.T[0:9]]

# plotting all features against each other
plt.figure(figsize=(10,10),dpi=100)

# Iterating through the features twice
for i in range(len(attributeNames)-1):
    for j in range(len(attributeNames)-1):
        
        # Defining the subplots
        plt.subplot(9,9, i*9 +j +1)

        # plotting the the features
        plt.plot(data[:,i], data[:,j], '.')

        # Setting a name for the columns
        if i == 8:
            plt.xlabel(attributeNames[j+1])
        #else:
        #    plt.xticks([])
        # Setting a name for the rows
        if j == 0:
            plt.ylabel(attributeNames[i+1])
        #else:
        #    plt.yticks([])

        # Set the x,y limits just above what is displayed
        plt.xlim(0,data[:,i].max()*1.1)
        plt.ylim(0,data[:,j].max()*1.1)


# Pakker subplot lidt tættere sammen
#plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

