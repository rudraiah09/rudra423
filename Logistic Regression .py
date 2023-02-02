#!/usr/bin/env python
# coding: utf-8

# # What is Clasification:
Classification in machine learning is the task of assigning a class label to a given input data point. 
The input data is often referred to as the feature vector and the class label is the output. 
The goal of the classifier is to learn a model that can accurately predict the class label for new, unseen data points.
There are many different types of classification algorithms, such as logistic regression, 
decision trees, and support vector machines, that can be used to solve classification problems.
# # What is Logistic Regression:
Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables
that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). 
It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.
Logistic Regression is similar to linear regression, but the curve is adapted to predict probabilities.
The method estimates the probability of the default class and the probability of the other class.
Logistic Regression can be used for various classification problems such as spam detection, tumor detection etc.Defination: Predict a Categorical dependent variable from number of Independent variable
# # What is find out in Logistic Regression:
Logistic Regression is used to find the relationship between a set of independent variables and a binary outcome variable.
It can be used to determine the probability of the outcome given the independent variables,
and to predict the outcome based on the values of the independent variables.
Logistic Regression can also be used to identify which independent variables have the most impact on the outcome,
and to understand how the independent variables are related to the outcome. Additionally,
Logistic Regression can be used to identify any interactions or 
non-linear relationships between the independent variables and the outcome.
# # when we have to do feature scaling in machine scaling? which situation  explain with example?
# Feature scaling is typically necessary in machine learning when the input features have different units of measurement or different scales. This is because many machine learning algorithms use distance-based metrics, such as Euclidean distance, to compare samples. Features with larger scales can dominate those with smaller scales, leading to poor performance or inaccurate results.

For example, consider a dataset with two features: "house size" measured in square feet and "price" measured in dollars. Without feature scaling, the algorithm may consider "house size" to be much more important than "price" because the values for "house size" are much larger. However, by scaling the features so that they have the same scale, the algorithm can properly weigh the importance of both features.

Common feature scaling techniques include standardization and normalization. Standardization scales the feature so that it has a mean of 0 and a standard deviation of 1, while normalization scales the feature so that it has a minimum value of 0 and a maximum value of 1.
# In[3]:


from IPython.display import Image
Image(filename = "C:/Users/thiru/Downloads/Rakesh/Logistic Regression.png", width = 600,height = 600)


# In[4]:


from IPython.display import Image
Image(filename = "C:/Users/thiru/Downloads/IMG_20230127_124941.jpg", width = 600,height = 600)


# # Import the Required Libraries:

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Load the data:

# In[2]:


data = pd.read_csv("C:/data Science/Machine Learning-A-Z-Codes-Datasets/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv")


# In[3]:


data


# In[4]:


data.isna().sum()


# In[5]:


data.isnull().sum()


# In[6]:


data.corr()


# In[7]:


sns.heatmap(data.corr(),annot = True)


# In[8]:


X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# # Splitting the dataset into Training and Testing:

# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# In[10]:


print(X_train)


# In[11]:


print(X_test)


# In[12]:


print(y_train)


# In[13]:


print(y_test)


# # Feature Scaling:

# In[14]:


from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


print(X_train)


# In[16]:


print(X_test)


# In[17]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)


# # Predict the test result:

# In[18]:


y_pred = classifier.predict(X_test)


# In[19]:


print(y_pred)


# In[20]:


print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# # predict the New Result:

# In[21]:


classifier.predict(sc.transform([[30,87000]]))


# # find confution_matrix

# In[22]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)


# # find accuracy score

# In[23]:


accuracy_score(y_test,y_pred)


# In[24]:


y_prob = classifier.predict_proba(X_test)


# In[25]:


print(y_prob)


# # Visualisation Train set result:

# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# # Visualising the test set Result:

# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

