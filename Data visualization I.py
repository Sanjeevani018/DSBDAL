#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset=sns.load_dataset('Titanic')
dataset.head()


# In[4]:


sns.distplot(x=dataset['age'],bins=10)


# In[8]:


sns.distplot(x=dataset['age'],bins=10,kde=False)


# In[9]:


sns.jointplot(x=dataset['age'],y=dataset['fare'],kind='scatter')


# In[10]:


sns.jointplot(x=dataset['age'],y=dataset['fare'],kind='hex')


# In[11]:


sns.rugplot(dataset['fare'])


# In[12]:


sns.barplot(x='sex',y='age',data=dataset,estimator=np.std)


# In[14]:


sns.countplot(x='sex',data=dataset)


# In[15]:


sns.boxplot(x='sex',y='age',data=dataset)


# In[17]:


sns.violinplot(x='sex',y='age',data=dataset,hue='survived')


# In[18]:


sns.stripplot(x='sex',y='age',data=dataset,jitter=False)


# In[19]:


sns.stripplot(x='sex',y='age',data=dataset,jitter=True)


# In[20]:


sns.swarmplot(x='sex',y='age',data=dataset,hue='survived')


# In[21]:


corr=dataset.corr()
sns.heatmap(corr)


# In[22]:


corr=dataset.corr()
sns.heatmap(corr,annot=True)


# In[27]:


sns.histplot(dataset['fare'], kde=False, bins=10)


# In[ ]:




