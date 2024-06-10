#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\APOORV PANDEY\Downloads\Salary_Data.csv")
data


# In[23]:


x = data.iloc[:,:-1].values
y =data.iloc[:,-1].values

    


# In[24]:


print(x)


# In[25]:


print(y)


# In[ ]:





# In[36]:


#splitting the data set into training and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state=0)


# In[27]:


#training the regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[28]:


#predicting the test results
y_pred = regressor.predict(x_test)


# 

# In[38]:


#visualising the training data
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# In[31]:


#visualizing the test data 
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs years of experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# In[39]:


#making a single prediction for a 12 years experience
print(regressor.predict([[14]]))

#this is how a model is used 


# In[34]:


print(regressor.coef_)
print(regressor.intercept_)


# In[ ]:




