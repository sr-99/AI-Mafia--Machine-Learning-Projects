#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


# In[64]:


dfx=pd.read_csv("Logistic_X_Train.csv")
dfy=pd.read_csv("Logistic_Y_Train.csv")
test=pd.read_csv("Logistic_X_Test.csv")
x=dfx.values
y=dfy.values
test=test.values
y=y.reshape((-1,))
print(x.shape,test.shape)
print(y.shape)


# In[68]:


def hypothesis(x,theta,b):
    hx= np.dot(x,theta)+b
    return sigmoid(hx)


def sigmoid(h):
    return 1.0/(1.0+np.exp(-1.0*h))


def llcost_func(x,y,theta,b):
    n = x.shape[0]
    error=0.0
    for i in range(n):
        hx=hypothesis(x[i],theta,b)
        error+=y[i]*np.log(hx)+(1-y[i])*np.log(1-hx)
    
    return error


def llcost_func(x,y,theta,b):
    n = x.shape[0]
    error=0.0
    for i in range(n):
        hx=hypothesis(x[i],theta,b)
        error+=y[i]*np.log(hx)+(1-y[i])*np.log(1-hx)
    
    return error


def gradient(x,y,theta,b):
    grad_b=0.0
    grad_w=np.zeros(theta.shape)
    n=x.shape[0]
    
    for i in range(n):
        hx=hypothesis(x[i],theta,b)
        grad_w+=(y[i]-hx)*x[i]
        grad_b+=(y[i]-hx)
        
    grad_w/=n
    grad_b/=n
    
    return [grad_w,grad_b]


def gradient_ascent(x,y,learning_rate=1):
    w = 2*np.random.random((x.shape[1],))
    b = 5*np.random.random()
    
    itr = 0
    max_itr = 500
    
    error_list = []
    
    
    while(itr<=max_itr):
        
        [grad_w,grad_b] = gradient(x,y,theta,b)
        e = llcost_func(x,y,theta,b)
        error_list.append(e)
        
        b=b+learning_rate*grad_b
        w=w+learning_rate*grad_w
        
        itr += 1
        
    return theta,b,error_list

def predict(x,theta,b):
    output=[]
    n=x.shape[0]
    
    for i in range(n):
        if(hypothesis(x[i],theta,b)<0.5):
            output.append(0)
        else:
            output.append(1)
    
    return output


# In[69]:


final_w,final_b,error_list=gradient_ascent(x,y)


# In[70]:


plt.plot(error_list)


# In[71]:


output_Y_test=predict(test,final_w,final_b)


# In[63]:


print(output_Y_test)


# In[ ]:





# In[ ]:




