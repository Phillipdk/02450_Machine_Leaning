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




## -------------Det sidste plot her virker ikke efter hensigten endnu.
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
# Import Data
#df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/USArrests.csv')
# Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(df[['PA500', 'Max IP', 'A/DA', 'I0']])  
# Plot
plt.figure(figsize=(14, 10), dpi= 80)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=cluster.labels_, cmap='tab10')  
# Encircle
def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

classnr = 'PA500'
classnr1 = 'A/DA'

# Draw polygon surrounding vertices    
encircle(df.loc[cluster.labels_ == 0, classnr], df.loc[cluster.labels_ == 0, classnr1], ec="k", fc="gold", alpha=0.2, linewidth=0)
#encircle(df.loc[cluster.labels_ == 1, classnr], df.loc[cluster.labels_ == 1, classnr1], ec="k", fc="tab:blue", alpha=0.2, linewidth=0)
#encircle(df.loc[cluster.labels_ == 2, classnr], df.loc[cluster.labels_ == 2, classnr1], ec="k", fc="tab:red", alpha=0.2, linewidth=0)
#encircle(df.loc[cluster.labels_ == 3, classnr], df.loc[cluster.labels_ == 3, classnr1], ec="k", fc="tab:green", alpha=0.2, linewidth=0)
#encircle(df.loc[cluster.labels_ == 4, classnr], df.loc[cluster.labels_ == 4, classnr1], ec="k", fc="tab:orange", alpha=0.2, linewidth=0)

# Decorations
plt.xlabel('Murder'); plt.xticks(fontsize=12)
plt.ylabel('Assault'); plt.yticks(fontsize=12)
plt.title('Agglomerative Clustering of USArrests (5 Groups)', fontsize=22)
plt.show()





"""
for i in range(len(attributeNames)-1):
    for j in range(len(attributeNames)-1):
        
        # Defining the subplots
        plt.subplot(9,9, i*9 +j +1)

        # plotting the the features
        #plt.plot(data[:,i], data[:,j], '.')
        sn.scatterplot(y=data[:,i], x=data[:,j], hue=data[:,0])

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


        
    
plt.figure(), plt.plot(data[1])





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

"""