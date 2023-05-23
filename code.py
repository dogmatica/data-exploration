#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# In[2]:


df = pd.read_csv('churn_clean.csv',dtype={'locationid':np.int64})


# In[3]:


data_slice = df[['Churn', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'Area']]


# In[4]:


data_slice


# In[5]:


ct_Marital = pd.crosstab(data_slice.Churn, data_slice.Marital, margins=True)
ct_Marital


# In[6]:


ct_Gender = pd.crosstab(data_slice.Churn, data_slice.Gender, margins=True)
ct_Gender


# In[7]:


ct_Area = pd.crosstab(data_slice.Churn, data_slice.Area, margins=True)
ct_Area


# In[8]:


obs = np.array([ct_Marital.iloc[0][0:5].values, ct_Marital.iloc[1][0:5].values])
stats.chi2_contingency(obs)[0:3]


# In[9]:


obs = np.array([ct_Gender.iloc[0][0:3].values, ct_Gender.iloc[1][0:3].values])
stats.chi2_contingency(obs)[0:3]


# In[10]:


obs = np.array([ct_Area.iloc[0][0:3].values, ct_Area.iloc[1][0:3].values])
stats.chi2_contingency(obs)[0:3]


# In[11]:


plt.hist(data_slice['Age'])


# In[12]:


plt.hist(data_slice['Income'])


# In[13]:


sns.countplot(x="Gender", data=data_slice)


# In[14]:


sns.countplot(x="Marital", data=data_slice)


# In[15]:


biv_cont_data = data_slice[['Age', 'Income']]
sns.scatterplot(x="Age", y="Income", data=biv_cont_data)


# In[16]:


biv_cat_data = data_slice[['Gender', 'Marital']]
sns.displot(biv_cat_data, x='Marital', hue='Gender', multiple='stack')


# In[ ]:




