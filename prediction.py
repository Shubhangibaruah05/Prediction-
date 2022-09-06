#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# SEPT 2022
# 
# DATA SCIENCE AND BUSINESS ANALYTICS
# 
# TASK 1: PREDICTION USING SUPERVISED ML
# 
# DESCRIPTION: PREDICT THE PERCENTAGE OF AN STUDENT BASED ON THE NO.OF STUDY HOURS
# 
# EXECUTED BY : SHUBHANGI BARUAH/subhangibaruah39@gmail.com
# 
# importing data

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#reading data from remote link

url ="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
ds =pd.read_csv(url)
print("Data imported successfully")

ds.head(25)


# In[4]:


ds.head()


# In[5]:


ds.tail()


# In[6]:


ds.shape


# In[7]:


ds.info()


# In[8]:


ds.describe()


# In[9]:


#to check any null or missing values
ds.isnull().sum()


# # data visualization

# In[14]:


#plotting the distribution of scores
plt.rcParams['figure.figsize']=[10,5]
ds.plot(x='Hours', y='Scores', style='*', color='red',markersize=10)
plt.title('hours vs percentage')
plt.xlabel('Hours studied')
plt.ylabel('percentage scored')
plt.grid()
plt.show()


# # Data preparation
# 
# TO DIVIDE THE DATA INTO ATTRIBUTES(INPUT) AND LABELS(OUTPUTS)

# In[51]:


#splitting the data using iloc function
x =ds.iloc[:, :1].values
y =ds.iloc[:, 1:].values


# In[52]:


x


# In[53]:


y


# In[54]:


#splitting the data into training and testing data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Training the algorithm

# In[55]:


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("training complete.")


# VISUAL DATA MODEL

# In[58]:


#plotting the regression line
line = regressor.coef_*x+regressor.intercept_


#plotting the test data
plt.rcParams['figure.figsize']=[10,5]
plt.scatter(x,y, color='purple')
plt.plot(x,line,color='green');
plt.xlabel=('percentage scored')
plt.ylabel=('hours studied')
plt.grid()
plt.show()


# In[26]:


plt.rcParams['figure.figsize']=[10,5]
plt.scatter(X_test,y_test, color='purple')
plt.plot(X,line,color='green');
plt.xlabel=('percentage scored')
plt.ylabel=('hours studied')
plt.grid()
plt.show()


# # Making predictions

# In[60]:



print(x_test)
y_pred = regressor.predict(x_test)


# In[62]:


y_pred


# In[30]:


y_test


# In[61]:


#comparing actual vs predicted
df=pd.DataFrame({'actual':[y_test],'predicted':[y_pred]})
df


# In[65]:


# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format([[hours]]))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluating the model

# In[64]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




