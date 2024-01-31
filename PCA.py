#!/usr/bin/env python
# coding: utf-8

# ## The curse of Dimensionality
# 
# Humans are bound by their perception of a maximum of three dimensions. We can’t comprehend shapes/graphs beyond three dimensions. Often, data scientists get datasets which have thousands of features. They give birth to two kinds of problems:
# 
# * **Increase in computation time:** Majority of the machine learning algorithms they rely on the calculation of distance for model building and as the number of dimensions increases it becomes more and more computation-intensive to create a model out of it. For example, if we have to calculate the distance between two points in just one dimension, like two points on the number line, we’ll just subtract the coordinate of one point from another and then take the magnitude:
# 
# Distance= $ x_1-x_2 $
# 
# What if we need to calculate the distance between two points in two dimensions?
# 
# The same formula translates to:
# Distance= $ \sqrt {(x_1-x_2)^2+(y_1-y_2)^2} $
# 
# What if we need to calculate the distance between two points in three dimensions?
# 
# The same formula translates to:
# Distance= $ \sqrt {(x_1-x_2)^2+(y_1-y_2)^2+(z_1-z_2)^2}$
# 
# And for N-dimensions, the formula becomes:
# Distance=$ \sqrt {(a_1-a_2)^2+(b_1-b_2)^2+(c_1-c_2)^2+…+(n_1-n_2)^2} $
# 
# This is the effort of calculating the distance between two points. Just imagine the number of calculations involved for all the data points involved.
# 
# One more point to consider is that as the number of dimension increases, points are going far away from each other. This means that any new point that comes when we are testing the model is going to be farther away from our training points. This leads to a less reliable model, and it makes our model overfitted to the training data.
# 
# 
# 
# * **Hard (or almost impossible) to visualise the relationship between features:** As stated above, humans can not comprehend things beyond three dimensions. So, if we have an n-dimensional dataset, the only solution left to us is to create either a 2-D or 3-D graph out of it. Let’s say for simplicity, we are creating 2-D graphs. Suppose we have 1000 features in the dataset. That results in a  total (1000*999)/2= 499500 combinations possible for creating the 2-D graph.
# 
# Is it humanly possible to analyse all those graphs to understand the relationship between the variables?
# 
# **The questions that we need to ask at this point are:**
# 
# * Are all the features really contributing to decision making?
# * Is there a way to come to the same conclusion using a lesser number of features?
# * Is there a way to combine features to create a new feature and drop the old ones?
# * Is there a way to remodel features in a way to make them visually comprehensible?
# 
# The answer to all the above questions is- _Dimensionality Reduction technique._
# 
# 
# 

# ## Principal Component Analysis: 
# The principal component analysis is an unsupervised machine learning algorithm used for feature selection using dimensionality reduction techniques. As the name suggests, it finds out the principal components from the data. PCA transforms and fits the data from a higher-dimensional space to a new, lower-dimensional subspace This results into an entirely new coordinate system of the points where the first axis corresponds to the first principal component that explains the most variance in the data.
# 
# **What are the principal components?**
# Principal components are the derived features which explain the maximum variance in the data. The first principal component explains the most variance, the 2nd a bit less and so on. Each of the new dimensions found using PCA is a linear combination of the old features.
# 
# Let's take the following example where the data is distributed like the diagram on the left:
# <img src="PCA_intro1.PNG" width="500">
# 
# 

# **what is the optimum number of Principal components needed?**
# 
# 

# #### Explained Variance Ratio
# 
# All of the above questions are answered using the *explained variance ratio*. It represents the amount of variance each principal component is able to explain.
# 
# For example, suppose if the square of distances of all the points from the origin that lie on PC1 is 50 and for the points on PC2 it’s 5.
# 
# EVR of PC1=$\frac{Distance of PC1 points}{( Distance of PC1 points+ Distance of PC2 points)}=\frac{50}{55}=0.91 $
# 
# EVR of PC2=$\frac{Distance of PC2 points}{( Distance of PC1 points+ Distance of PC2 points)}=\frac{5}{55}=0.09 $
# 
# 
# Thus PC1 explains 91% of the variance of data. Whereas, PC2 only explains 9% of the variance. Hence we can use only PC1 as the input for our model as it explains the majority of the variance.
# 
# In a real-life scenario, this problem is solved using the **Scree Plots**

# ## Scree Plots:
# Scree plots are the graphs that convey how much variance is explained by corresponding Principal components. 
# <img src="scree.PNG" width="500">
# 
# As shown in the given diagram, around 75 principal components explain approximately 90 % of the variance. Hence, 75 can be a good choice based on the scenario

# ## Python Implementation

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# we are using the free glass datset.
# The objective is to tell the type of glass based on amount of other elements present.
data = pd.read_csv('glass.data')


# In[4]:


data.head()


# In[5]:


data.Class.unique() #3 6 class classification


# In[ ]:


## Basic checks


# In[6]:


data.info()


# In[7]:


data.describe()


# In[ ]:


## Exploratory Data Analysis


# ## Data Preprocessing

# In[8]:


## CHecking missing values
data.isnull().sum()


# In[9]:


## creating X and y
X=data.drop(labels=['index','Class'], axis=1)


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(X)


# In[11]:


## creating new dataframe
df=pd.DataFrame(data=scaled_data, columns= X.columns)


# In[12]:


df.head()


# In[13]:


## getting the optimal number of pca
from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(df)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()


# From the diagram above, it can be seen that 4 principal components explain almost 90% of the variance in data and 5 principal components explain around 95% of the variance in data.
# 
# So, instead of giving all the columns as input, we’d only feed these 4 principal components of the data to the machine learning algorithm and we’d obtain a similar result.

# In[14]:


pca = PCA(n_components=4)
new_data = pca.fit_transform(df)
# This will be the new data fed to the algorithm.
principal_Df = pd.DataFrame(data = new_data
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])


# In[15]:


## new pca dataframe
principal_Df.head()


# Here, we see that earlier we had 9 columns in the data earlier. Now with the help of Scree plot and PCA, we have reduced the number of features to be used for model building to 4. This is the advantage of PCA. _It drastically reduces the number of features, thereby considerably reducing the training time for the model._

# ### Visualizing the Principal components
# 
# As humans can only perceive 3dimensions, we’ll take a dataset with less than 4 dimensions. 
# 
# 

# In[16]:


np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T
plt.plot(X[:, 0], X[:, 1], 'o')
plt.axis('equal');


# In[ ]:


## task: create 2 models where in first model with all fetaures and second model with 4 pca features and check the performance


# **Pros of PCA:**
# 
# - Correlated features are removed.
# - Model training time is reduced.
# - Overfitting is reduced.
# - Helps in better visualizations
# - Ability to handle noise
# 
# **Cons of PCA**
# - The resultant principal components are less interpretable than the original data
# - Can lead to information loss if the explained variance threshold is not considered appropriately.
# 
# 

# ### Conclusion
# From all the explanations above, we can conclude that PCA is a very powerful technique for reducing the dimensions of the data, projecting the data from a higher dimension to a lower dimension, helps in data visualization, helps in data compression and most of all increases the model training speed drastically by decreasing the number of variables involved in computation.

# https://i.stack.imgur.com/Q7HIP.gif
