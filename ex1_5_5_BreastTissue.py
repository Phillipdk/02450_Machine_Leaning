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


for pos, val in enumerate(data):
    print(pos, "  ", val[4])
    if val[4]>50000:
        print("delete this one")



for pos, val in enumerate(data):
    print(pos, "  ", val[4])
    if val[4]>50000:
        print("delete this one")
        data = np.delete(data, pos, 0)

np.savetxt("ordnet_data.csv", data, delimiter=",")

#[(plt.figure(), plt.plot(row)) for row in data.T[0:9]]


#%%


plt.figure(figsize=(10,10),dpi=100)
for m1 in range(9): # number of features
    for m2 in range(9): # number of features
        plt.subplot(9, 9, m1*9 + m2 + 1) # iterates over 9x9 subplots
        for i in range(len(tissue_types)):
            plt.plot(np.array(data[:,m2]), np.array(data[:,m1]), '.')
            #class_mask = (y==c)
            #plt.plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            #if m1==M-1:
            #    xlabel(attributeNames1[m2][0:13])
            #else:
            #    xticks([])
            #if m2==0:
            #    ylabel(attributeNames1[m1][0:13])
            #else:
            #    yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
#plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
#plt.legend(classNames)
#plt.legend(classNames, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



