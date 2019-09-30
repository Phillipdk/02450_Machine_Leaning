## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Load file in
file_path = './BreastTissue.xls'
breast_data = pd.read_excel(file_path, sheet_name = "Data", index_col=0)

sns.set()
scatterplot_data_nr1 = breast_data
g = sns.PairGrid(scatterplot_data_nr1)
g = g.map(plt.scatter)

#sns.clustermap(scatterplot_data_nr1[1:, :])

#del K, i, tiss, tissue_names, file_path

# Show the attributes 
attributeNames = np.array(breast_data.columns)

# Isolate the class types
tissue_names = np.array(breast_data.Class)
#breast_data = breast_data.drop(['Class'], axis=1)

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
        scatterplot_data_nr1 = np.delete(scatterplot_data_nr1, pos, 0)

# Save to .txt file
#np.savetxt("ordnet_data.csv", data, delimiter=",")

#i = 1, j = 2


#sns.scatterplot(y=data[:,2], x=data[:,3], hue=data[:,0])

#sns.relplot(x=data[:,1], y=data[:, 2], hue=data[:, 0], data=scatterplot_data_nr1)

"""
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

fig = plt.figure()
ax = Axes3D(fig)
xs = scatterplot_data_nr1['I0']
ys = scatterplot_data_nr1['DR'],
zs = scatterplot_data_nr1['P']
ax.scatter(xs, ys, zs, s=50, alpha=1, edgecolors='b')
ax.set_xlabel('A/DA')
ax.set_ylabel('HFS')
ax.set_zlabel('P')
plt.show()
"""




df = scatterplot_data_nr1
# Plot corrolellogram se https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/#1.-Scatter-plot
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
# Decorations
plt.title('Correlogram', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#regression pairplot
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="reg", hue="Class")
plt.show()



#Density plots
dataClasses = "I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"
for nameOfClass in dataClasses:
    plt.figure(figsize=(16,10), dpi= 80)
    sns.kdeplot(df.loc[df['Class'] == 'car', nameOfClass], shade=True, color="g", label="car", alpha=.7)
    sns.kdeplot(df.loc[df['Class'] == 'fad', nameOfClass], shade=True, color="deeppink", label="other", alpha=.7)
    sns.kdeplot(df.loc[df['Class'] == 'mas', nameOfClass], shade=True, color="deeppink", label="other", alpha=.7)
    sns.kdeplot(df.loc[df['Class'] == 'gla', nameOfClass], shade=True, color="deeppink", label="other", alpha=.7)
    sns.kdeplot(df.loc[df['Class'] == 'con', nameOfClass], shade=True, color="deeppink", label="other", alpha=.7)
    sns.kdeplot(df.loc[df['Class'] == 'adi', nameOfClass], shade=True, color="deeppink", label="other", alpha=.7)
    # Decoration
    plt.title('Density Plot of '+nameOfClass, fontsize=22)
    plt.legend()
    plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Setup
rng = np.random.RandomState(0)  # Seed RNG for replicability
n = 100  # Number of samples to draw

# Generate data
x = rng.normal(size=n)  # Sample 1: X ~ N(0, 1)
y = rng.standard_t(df=5, size=n)  # Sample 2: Y ~ t(5)

# Quantile-quantile plot
plt.figure()
plt.scatter(np.sort(x), np.sort(y))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.close()