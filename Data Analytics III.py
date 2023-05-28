#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
iris = pd.read_csv(csv_url, names = col_names)
iris.head()


# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris['Species'] = le.fit_transform(iris['Species'])
iris.head()


# In[9]:


iris.isnull().sum()


# In[10]:


x = iris.iloc[:,:4]
y = iris['Species']
x


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[13]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)


# In[14]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# In[15]:


accuracy = accuracy_score(y_test,y_pred)
precision =precision_score(y_test, y_pred,average='micro')
recall = recall_score(y_test, y_pred,average='micro')


print("Accuracy:- ",accuracy)
print("Precision:- ",precision)
print("Recall:- ",recall)


# In[16]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[17]:


from sklearn.metrics import ConfusionMatrixDisplay
cmD = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = ["Iris-setosa","Iris-versicolor","Iris-virginica"])


# In[18]:


cmD.plot()


# In[ ]:




