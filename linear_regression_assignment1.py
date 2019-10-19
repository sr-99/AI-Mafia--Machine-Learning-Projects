#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#to read the input data from the csv files

dfx = pd.read_csv('./Training Data/Linear_X_Train.csv')
dfy = pd.read_csv('./Training Data/Linear_Y_Train.csv')

x = dfx.values
y = dfy.values

#reshaping the data file 

x = x.reshape((-1,))
y = y.reshape((-1,))

print(x.shape)
print(y.shape)

#plotting the data

plt.scatter(x,y,color="green")
plt.show()


#plotting after normalizing the x

X = (x-x.mean())/x.std()
Y = y
plt.scatter(X,Y,color="red")
plt.show()


    


# In[83]:



def hypothesis(x,theta):
    return theta[0] + theta[1]*x


def error(X,Y,theta):
    
    m = X.shape[0]
    error = 0
    
    for i in range(m):
        hx = hypothesis(X[i],theta)
        error += (hx-Y[i])**2
        
    return error

def gradient(X,Y,theta):
    
    grad = np.zeros((2,))
    m = X.shape[0]

    for i in range(m):
        hx = hypothesis(X[i],theta)
        grad[0] +=  (hx-Y[i])
        grad[1] += (hx-Y[i])*X[i]
        
    return grad
    

def gradientDescent(X,Y,learning_rate=0.001):
    
    theta = np.array([-2.0,0.0])
    
    itr = 0
    max_itr = 100
    
    error_list = []
    theta_list = []
    
    while(itr<=max_itr):
        grad = gradient(X,Y,theta)
        e = error(X,Y,theta)
        error_list.append(e)
        
        theta_list.append((theta[0],theta[1]))
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        
        
        itr += 1
        
    
    return theta,error_list,theta_list

    


# In[86]:


final_theta, error_list,theta_list = gradientDescent(X,Y) 
plt.plot(error_list)
plt.show()
print(final_theta)
testing=pd.read_csv('./Testing Data/Linear_X_Test.csv')
test=testing.values
print(test)
new= (test-test.mean())/test.std()
print(new)


# In[88]:


plt.scatter(X,Y,label='Training Data')
plt.show()
plt.scatter(X,Y,label='Training Data')
plt.plot(new,hypothesis(new,final_theta),color='orange',label="Prediction")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[79]:





# In[ ]:





# In[81]:



  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




