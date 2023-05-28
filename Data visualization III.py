#!/usr/bin/env python
# coding: utf-8

# In[6]:


import seaborn as sns 
dataset=sns.load_dataset('iris')
dataset.head()


# In[8]:


import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(16,9))
sns.histplot(dataset['sepal_length'],ax=axes[0,0])
sns.histplot(dataset['sepal_length'],ax=axes[0,1])
sns.histplot(dataset['sepal_length'],ax=axes[1,0])
sns.histplot(dataset['sepal_length'],ax=axes[1,1])


# In[12]:


fig,axes=plt.subplots(2,2,figsize=(16,9))
sns.boxplot(y='petal_length',x='species',data=dataset,ax=axes[0,0])
sns.boxplot(y='petal_width',x='species',data=dataset,ax=axes[0,1])
sns.boxplot(y='sepal_length',x='species',data=dataset,ax=axes[1,0])
sns.boxplot(y='sepal_length',x='species',data=dataset,ax=axes[1,1])


# In[ ]:





# In[ ]:




