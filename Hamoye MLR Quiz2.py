#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/KEMI- ITB CONCRETE/Desktop/energydata_complete.csv')
df


# In[2]:


df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[9]:


df.columns


# In[11]:


df.drop(columns=['date','lights'], inplace=True)


# In[12]:


df.columns


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler


# In[14]:


normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
normalised_df


# In[15]:


df.head()


# In[17]:


normalised_df.head()


# In[20]:


features_df = normalised_df.drop(['Appliances'], axis=1)
appliances_target = normalised_df['Appliances']


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(features_df, appliances_target, test_size=0.3, random_state=42)


# In[23]:


linear_model.fit(x_train, y_train)


# In[24]:


predicted_values = linear_model.predict(x_test)
predicted_values


# In[25]:


#Root mean square error
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[26]:


#R-Squared
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 3)


# In[27]:


# Residual Sum of Squares
import numpy as np
rss = np.sum(np.square(y_test - predicted_values))
round(rss, 3)


# In[28]:


#MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae,3)


# In[29]:


#this function returns the weight of every feature
def get_weights_df(model, feat, col_name):
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[30]:


#RidgeRegression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(x_train, y_train)


# In[31]:


#Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[32]:


linear_model_weights = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, x_train,'Lasso_Weight')
lasso_weights_df


# In[33]:


final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='features')
final_weights


# In[ ]:




