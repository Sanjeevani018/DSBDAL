#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.datasets import load_iris
import pandas as pd
data=load_iris()
iris=pd.DataFrame(data=data.data,columns=data.feature_names)
iris['Species']=data.target
print(iris)


# In[29]:


iris.shape


# In[30]:


iris.dtypes


# In[47]:


iris.sort_values(by="sepal length (cm)", ascending=False)


# In[46]:


iris.loc[11:20,["sepal length (cm)","Species"]]


# In[48]:


iris.rename(columns={"Species":"Type"} ,inplace=True)
iris


# In[ ]:




