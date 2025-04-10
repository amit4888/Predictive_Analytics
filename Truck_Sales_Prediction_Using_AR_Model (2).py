#!/usr/bin/env python
# coding: utf-8

# ![Screenshot%202025-04-10%20054524.png](attachment:Screenshot%202025-04-10%20054524.png)

# # Import Essential Library

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error

# Import Warnings Library
import warnings
warnings.filterwarnings("ignore")


# # Data Acquisition

# In[3]:


df = pd.read_csv("Truck_sales.csv")


# In[4]:


# Shape of the dataset
df.shape


# In[5]:


# Display First 5 rows of the dataset
df.head()


# In[6]:


# now, this data we'll convert into time series
df = pd.read_csv("Truck_sales.csv",parse_dates = ['Month-Year'],index_col = 'Month-Year')


# In[7]:


df.head()


# In[9]:


df.plot()

plt.title("Truck Sales Over Time")  

plt.xlabel("Date")  
plt.ylabel("Number of Trucks Sold")   

# Show the plot
plt.show()


# In[23]:


# Suppose df is your DataFrame and 'Sales' is your time series column
df['Number_Trucks_Sold'] = df['Number_Trucks_Sold'] - df['Number_Trucks_Sold'].shift(1)

# Or using .diff() directly
df['Number_Trucks_Sold'] = df['Number_Trucks_Sold'].diff()

# Drop NaN from first row (since it has no previous value)
df.dropna(inplace=True)


# In[10]:


from statsmodels.tsa.stattools import adfuller

dftest = adfuller(df['Number_Trucks_Sold'], autolag = 'AIC')

print("1. ADF : ",dftest[0])
print("2. P-Value :", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Usd for ADF Regression andd critical values Calculation : ", dftest[3])
print("5. Critical Values : ",)
for key, val in dftest[4].items():
    print("\t",key, ": ",val)


# In[11]:


from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

pacf=plot_pacf(df['Number_Trucks_Sold'],lags=25)
acf=plot_acf(df['Number_Trucks_Sold'],lags=25)


# In[17]:


# train=X[:len(X)-7]
# test=X[len(X)-7:]


# In[18]:


# model=AutoReg(train,lags=10)


# In[19]:


# model_fit = model.fit()
# print(model_fit.summary())


# In[20]:


# pred = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)


# In[21]:


# import matplotlib.pyplot as plt

# # Plot predictions vs actual values
# plt.figure(figsize=(6,6))
# plt.plot(pred, label='Prediction', linestyle='dashed')
# plt.plot(test, label='Actual', linestyle='solid', color='red')
# plt.legend()
# plt.title('AutoRegression Model - Predicted vs Actual')
# plt.show() 

# # Print predicted values
# print(pred)


# In[ ]:




