#!/usr/bin/env python
# coding: utf-8

# # Numpy Exercise

# #### 1. Import the numpy package under the name  (★☆☆)

# In[1]:


import numpy as np


# #### 2. Print the numpy version and the configuration  (★☆☆)

# In[2]:


import numpy as np
print(np.__version__)


# #### 3. Create a null vector of size 10 (★☆☆)

# In[4]:


import numpy as np
n=np.zeros(10)
print(n)


# #### 4.  How to find the memory size of any array (★☆☆)

# In[6]:


import numpy as np
from sys import getsizeof
n=np.array(10)
print(getsizeof(n))


# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

# In[ ]:





# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

# In[7]:


import numpy as np 
n=np.zeros(10)
print(n)
n[4]=1
print(n)


# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

# In[9]:


import numpy as np
array=np.arange(10,49)
print(array)


# #### 8.  Reverse a vector (first element becomes last) (★☆☆)

# In[11]:


import numpy as np
array=np.array([1,2,3,4,5,6])
print("before reversing",end=" ")
print(array)
reverse=array[: :-1]
print("after reversing",end="")
print(reverse)


# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

# In[15]:


import numpy as np
n=np.arange(0,9)
print(n)
new=n.reshape(3,3)
print(new)


# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

# In[22]:


import numpy as np
n=np.array([1,2,0,0,4,0])
for i in range(0,n.size):
     if n[i]!=0:
        print(i) 


# #### 11. Create a 3x3 identity matrix (★☆☆)

# In[24]:


import numpy as np
n=np.eye(3,3)
print(n)


# #### 12. Create a 3x3x3 array with random values (★☆☆)

# In[3]:


import numpy as np
n=np.arange(27)
print(n)
np.random.seed(1)
np.random.shuffle(n)
print(n)
new=n.reshape(3,3,3)
print(new)


# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

# In[9]:


import numpy as np
n=np.arange(0,100)
print(n)
np.random.seed(2)
np.random.shuffle(n)
new=n.reshape(10,10)
print(np.max(n))
print(np.min(n))


# #### 14. Create a random vector of size 30 and find the mean value (★☆☆)

# In[12]:


import numpy as np
n=np.arange(55,85)
print(n)
np.random.seed(2)
np.random.shuffle(n)
print(n)
print(np.mean(n))


# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

# In[21]:


import numpy as np
n=np.array([[1,1,1],[1,0,1],[1,1,1]])
new=n.reshape(3,3)
print(new)


# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# In[ ]:





# #### 17. What is the result of the following expression? (★☆☆)

# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# np.nan in set([np.nan])
# 0.3 == 3 * 0.1
# ```

# In[22]:





# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

# In[ ]:





# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

# In[25]:


import numpy as np
n=np.ones((3,3))
print ("checker board pattern")
x=np.zeros((8,8),dtype=int)
x[1::2,::2]=1
x[::2,1::2]=1
print(x)


# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

# In[ ]:


import numpy as np
n=n.arange(336)
item=n[100]
new=n.reshape(6,7,8)
cond=item


# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

# In[28]:


import numpy as np
n=8
arr=([0,1])
print(np.tile(arr,(n,n//2)))


# #### 22. Normalize a 5x5 random matrix (★☆☆)

# In[34]:


import numpy as np
n=np.random.random((5,5))
print(n)
xmax,xmin=n.max(),n.min()
n=(n-xmin)/(xmax-xmin)
print(n)


# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# In[37]:


import numpy as np
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])


# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

# In[38]:


import numpy as np
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)


# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

# In[35]:


Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)


# #### 26. What is the output of the following script? (★☆☆)

# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# In[29]:


9 10


# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# In[ ]:


z**z


# #### 28. What are the result of the following expressions?

# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# In[ ]:


error


# #### 29. How to round away from zero a float array ? (★☆☆)

# In[33]:


import numpy as np
n=np.arange((10),dtype=float)
print(n)


# #### 30. How to find common values between two arrays? (★☆☆)

# In[39]:


import numpy as np
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# In[ ]:





# #### 32. Is the following expressions true? (★☆☆)

# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# In[ ]:





# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

# In[ ]:





# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

# In[ ]:





# 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

# In[40]:


import numpy as np
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)


# #### 36. Extract the integer part of a random array using 5 different methods (★★☆)

# In[42]:


import numpy as np
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)


# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

# In[41]:


import numpy as np
n= np.zeros((5,5))
n += np.arange(5)
print(n)


# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

# In[ ]:


import numpy as np


# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

# In[ ]:


import numpy as np


# #### 40. Create a random vector of size 10 and sort it (★★☆)

# In[43]:


import numpy as np
Z = np.random.random(10)
Z.sort()
print(Z)


# #### 41. How to sum a small array faster than np.sum? (★★☆)

# In[44]:


import numpy as np
n = np.arange(10)
np.add.reduce(n)


# #### 42. Consider two random array A and B, check if they are equal (★★☆)

# In[45]:


import numpy as np
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)


equal = np.array_equal(A,B)
print(equal)


# #### 43. Make an array immutable (read-only) (★★☆)

# In[46]:


import numpy as np
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1


# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

# In[ ]:


import numpy as np


# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

# In[47]:


import numpy as np
n = np.random.random(10)
n[n.argmax()] = 0
print(n)


# #### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

# In[48]:


import numpy as np
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)


# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

# In[ ]:


import numpy as np


# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

# In[ ]:


import numpy as np


# #### 49. How to print all the values of an array? (★★☆)

# In[ ]:


import numpy as np


# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

# In[49]:


import numpy as np
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])


# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

# In[ ]:





# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

# In[ ]:





# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

# In[50]:


import numpy as np
x = np.arange(10, dtype=np.float32)
x= x.astype(np.int32, copy=False)
print(x)


# #### 54. How to read the following file? (★★☆)

# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# In[ ]:





# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

# In[ ]:





# #### 56. Generate a generic 2D Gaussian-like array (★★☆)

# In[ ]:





# #### 57. How to randomly place p elements in a 2D array? (★★☆)

# In[ ]:


import numpy as np


# #### 58. Subtract the mean of each row of a matrix (★★☆)

# In[ ]:


import numpy as np


# #### 59. How to sort an array by the nth column? (★★☆)

# In[51]:


import numpy as np
Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])


# #### 60. How to tell if a given 2D array has null columns? (★★☆)

# In[ ]:


import numpy as np


# #### 61. Find the nearest value from a given value in an array (★★☆)

# In[ ]:





# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

# In[ ]:





# #### 63. Create an array class that has a name attribute (★★☆)

# In[ ]:





# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

# In[ ]:





# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

# In[ ]:





# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

# In[ ]:





# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

# In[ ]:





# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

# In[ ]:





# #### 69. How to get the diagonal of a dot product? (★★★)

# In[ ]:





# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

# In[ ]:





# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

# In[ ]:





# #### 72. How to swap two rows of an array? (★★★)

# In[ ]:





# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

# In[ ]:





# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

# In[ ]:





# #### 75. How to compute averages using a sliding window over an array? (★★★)

# In[ ]:





# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)

# In[ ]:





# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

# In[ ]:





# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# In[ ]:





# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

# In[ ]:





# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

# In[ ]:





# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)

# In[ ]:





# #### 82. Compute a matrix rank (★★★)

# In[ ]:





# #### 83. How to find the most frequent value in an array?

# In[ ]:





# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

# In[ ]:





# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)

# In[ ]:





# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

# In[ ]:





# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

# In[ ]:





# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# In[ ]:





# #### 89. How to get the n largest values of an array (★★★)

# In[ ]:





# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

# In[ ]:





# #### 91. How to create a record array from a regular array? (★★★)

# In[ ]:





# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

# In[ ]:





# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

# In[ ]:





# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

# In[ ]:





# #### 95. Convert a vector of ints into a matrix binary representation (★★★)

# In[ ]:





# #### 96. Given a two dimensional array, how to extract unique rows? (★★★)

# In[ ]:





# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

# In[ ]:





# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

# In[ ]:





# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

# In[ ]:





# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

# In[ ]:




