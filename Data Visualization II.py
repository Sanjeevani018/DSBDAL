#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns


# In[3]:


9dataset=sns.load_dataset('titanic')
dataset.head()


# In[4]:


sns.boxplot(x='sex',y='age',data=dataset)


# In[5]:


sns.boxplot(x='sex',y='age',data=dataset,hue='survived')


# In[ ]:


observation=more younger male survived than elder and more elder female survived than younger

