#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/KEMI- ITB CONCRETE/Desktop/energydata_complete.csv')
df


# In[44]:


x = df[['T2']]
y = df[['T6']]


# In[63]:


#split dataset into testing and training dataset.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[64]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)


# In[49]:


simple_linear_reg_df = df[['Appliances']].sample(15, random_state=42)
simple_linear_reg_df


# In[50]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler


# # Penalization Methods

# In[ ]:


#RidgeRegression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(x_train, y_train)


# In[ ]:


#Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[60]:


#this function returns the weight of every feature
def get_weights_df(model, feat, col_name):
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[52]:


predicted_values = linear_model.predict(x_test)
predicted_values


# In[67]:


#R-Squared
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 3)


# In[ ]:




