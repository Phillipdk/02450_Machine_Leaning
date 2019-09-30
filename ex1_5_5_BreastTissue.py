## exercise 1.5.5
import numpy as np
import pandas as pd

# Load file in
file_path = './BreastTissue.xls'
breast_data = pd.read_excel(file_path, sheet_name = "Data", index_col=0)
del file_path

#sns.pairplot(breast_data, hue="Class")
#plt.show()

# Save the attribute_names in a list: ['Class', 'IO', ..., 'P']
# len(attributesNames) = 10
attributeNames = np.array(breast_data.columns)

# Isolate the class types
# len(tissue_names) = 106
tissue_names = np.array(breast_data.Class)
# Removes the column with class definitions as prep for 1-out-of-K
breast_data_M = np.array(breast_data.drop(['Class'], axis=1))

# Lav 1-out-of-K coding
tissue_types = []
[tissue_types.append(elem) for elem in tissue_names if elem not in tissue_types]

K = np.zeros((len(breast_data_M), len(tissue_types)))

for i in range(len(tissue_names)):
    K[i][tissue_types.index(tissue_names[i])] = 1
del i

# Slå 1-out-of-K sammen data
data = np.concatenate((breast_data_M, K), axis=1)

# Remove outliers 
for pos, val in enumerate(data):
    if val[4]>50000:
        data = np.delete(data, pos, 0)
del pos, val

# Mean prep for PCA
for i in range(len(breast_data_M[0])):
    mean = sum(breast_data_M[:,i]) / len(breast_data_M[:,i])
    breast_data_M[:,i] -= mean
del i

# PCA
U, s, VT = np.linalg.svd(breast_data_M)
V = VT.T 
S =  [elem**2/sum([elem2**2 for elem2 in s]) for elem in s]

# Save to .txt file
np.savetxt("ordnet_data.csv", data, delimiter=",")