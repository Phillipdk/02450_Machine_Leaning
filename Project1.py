# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:40:46 2019

@author: Babro
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'BreastTissue.xls'

data = pd.read_excel(file_path, sep='\t', header=0, sheet_name="Data")

attributeNames = np.asanyarray(data.columns)

#print(attributeNames)


raw_data = data.get_values() 
cols = range(2, 11) 
X = raw_data[:, cols]

#print(X)

attributeNames1 = np.asarray(data.columns[cols])
#print(attributeNames1)

#entries=data.sort(['i','j','ColumnA','ColumnB'])
#attributeNames2 = data.row_values(rowx=0, start_colx=2, end_colx=8)
#

# Extract class names to python list, then encode with integers (dict) just as 
# we did previously. The class labels are in the 5th column, in the rows 2 to 
# and up to 151:
classLabels = raw_data[:,1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
# Extract vector y, convert to NumPy array
y = np.array([classDict[cl] for cl in classLabels])


# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
list = [0,1,2,3,4,5]
#new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)


# %% Classification problem
# The current variables X and y represent a classification problem, in
# which a machine learning model will use the sepal and petal dimesions
# (stored in the matrix X) to predict the class (species of Iris, stored in
# the variable y). A relevant figure for this classification problem could
# for instance be one that shows how the classes are distributed based on
# two attributes in matrix X:
X_c = X.copy();
y_c = y.copy();
attributeNames_c = attributeNames.copy();
i = 1; j = 2;
color = ['r','g', 'b','cyan','purple','orange']
plt.title('BreastTissue  classification problem')
for c in range(len(classNames)):
    idx = y_c == c
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()




#%% Regression problem
# Since the variable we wish to predict is petal length,
# petal length cannot any longer be in the data matrix X.
# The first thing we do is store all the information we have in the
# other format in one data matrix:
data = np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
# We need to do expand_dims to y_c for the dimensions of X_c and y_c to fit.
#print(data)


# We know that the petal length corresponds to the third column in the data
# matrix (see attributeNames), and therefore our new y variable is:
y_r = data[:, 2]

# Similarly, our new X matrix is all the other information but without the 
# petal length (since it's now the y variable):
X_r = data[:, 0:11]
#print(X_r)


# Since the iris class information (which is now the last column in X_r) is a
# categorical variable, we will do a one-out-of-K encoding of the variable:
species = np.array(X_r[:,-1], dtype=int).T
K = species.max()+1
species_encoding = np.zeros((species.size, K))
species_encoding[np.arange(species.size), species] = 1
# The encoded information is now a 150x3 matrix. This corresponds to 150
# observations, and 3 possible species. For each observation, the matrix
# has a row, and each row has two 0s and a single 1. The placement of the 1
# specifies which of the three Iris species the observations was.

# We need to replace the last column in X (which was the not encoded
# version of the species data) with the encoded version:
X_r = np.concatenate( (X_r[:, 0:11], species_encoding), axis=1) 

#print(X_r)

# Now, X is of size 150x6 corresponding to the three measurements of the
# Iris that are not the petal length as well as the three variables that
# specifies whether or not a given observations is or isn't a certain type.
# We need to update the attribute names and store the petal length name 
# as the name of the target variable for a regression:
targetName_r = attributeNames_c[2]
attributeNames_r = np.concatenate((attributeNames_c[3:11], classNames), axis=0)

#print(attributeNames_c)
#print(attributeNames_r)



# Lastly, we update M, since we now have more attributes:
N,M = X_r.shape

# A relevant figure for this regression problem could
# for instance be one that shows how the target, that is the petal length,
# changes with one of the predictors in X:
i = 1  
plt.title('BreastTissue  Regression problem')
plt.plot(X_r[:, i], y_r, 'o')
plt.xlabel(attributeNames_r[i]);
plt.ylabel(targetName_r);
# Consider if you see a relationship between the predictor variable on the
# x-axis (the variable from X) and the target variable on the y-axis (the
# variable y). Could you draw a straight line through the data points for
# any of the attributes (choose different i)? 
# Note that, when i is 3, 4, or 5, the x-axis is based on a binary 
# variable, in which case a scatter plot is not as such the best option for 
# visulizing the information. 









# %%
#from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, show)

plt.figure(figsize=(10,6),dpi=100)

for i in range(12):
    plt.subplot(3,4,i+1) 
    plt.boxplot(X[:,c],vert=False)
    #title('Class: {0}'.format(classNames[c]))
    plt.title(''+attributeNames1[c][0:40])
    
    plt.yticks(range(1), [a[:9] for a in attributeNames[c]])
    
    y_up = X[:,c].max()+(X[:,c].max()-X[:,c].min())*0.1; y_down = X[:,c].min()-(X[:,c].max()-X[:,c].min())*0.1
    plt.ylim(y_down, y_up)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)
show()


# %%
 

df = pd.DataFrame(X[:,0:9], columns=attributeNames1[0:])
desc = df.describe()
desc

fig = plt.subplots(1, 9, figsize=(16,5)) #1 row, 7 cols
fig[0].tight_layout()
fliers = []
for sp, col in zip(fig[1], desc.columns):
    fliers.append(sp.boxplot(df[col], sym="k."))
    sp.set_xticks([])
    sp.set_title(col)
    sp.set_title(col)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.4)
plt.show()
# %%

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xticks, yticks, xlabel, ylabel, show
from scipy.linalg import svd


M=9

figure(figsize=(10,10),dpi=100)
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames1[m2][0:13])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames1[m1][0:13])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
legend(classNames)
legend(classNames, loc='center left', bbox_to_anchor=(1, 0.5))
show()

