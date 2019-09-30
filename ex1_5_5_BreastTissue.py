## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import scipy as sci

# Load file in
file_path = './BreastTissue.xls'
breast_data = pd.read_excel(file_path, sheet_name = "Data", index_col=0)

sns.pairplot(breast_data, hue="Class")
plt.show()

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

# Slå 1-out-of-K sammen data
data = np.concatenate((breast_data_M, K), axis=1)

# Remove outliers 
for pos, val in enumerate(data):
    if val[4]>50000:
        data = np.delete(data, pos, 0)

# Save to .txt file
np.savetxt("ordnet_data.csv", data, delimiter=",")

sns.scatterplot(y=data[:,2], x=data[:,3], hue=data[:,0])

sns.relplot(x=data[:,1], y=data[:, 2], hue=data[:, 0], data=scatterplot_data_nr1)

sns.set()
g = sns.PairGrid(scatterplot_data_nr1, hue="Class")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()



plt.scatter(x = scatterplot_data_nr1['I0'], 
            y = scatterplot_data_nr1['DR'], 
            s = scatterplot_data_nr1['P'], # <== 😀 Look here!
            alpha=0.4, 
            edgecolors='w'
            )

plt.xlabel('DATA1')
plt.ylabel('DATA2')
plt.title('DATA3', y=1.05)
