#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[ ]:


import numpy as np


# 2. Create a null vector of size 10 

# In[4]:


import numpy as np
x = np.zeros(10)
print(x)


# 3. Create a vector with values ranging from 10 to 49

# In[11]:



v = np.arange(10,49)
print(v)


# 4. Find the shape of previous array in question 3

# In[13]:


v.shape


# 5. Print the type of the previous array in question 3

# In[14]:


type(v)


# 6. Print the numpy version and the configuration
# 

# In[15]:


print(np.__version__)


# In[19]:


print(np.__config__)


# 7. Print the dimension of the array in question 3
# 

# In[20]:


v.ndim


# 8. Create a boolean array with all the True values

# In[23]:


bool_arr = np.array([1, 0.5, 2, 4,], dtype=bool)
print(bool_arr)


# 9. Create a two dimensional array
# 
# 
# 

# In[27]:


arr1 = np.array([[1,4,7] , [4,9,3]])


# 10. Create a three dimensional array
# 
# 

# arr2 = np.array

# In[4]:


import numpy as np
array = np.arange(27).reshape(3,3,3)
array


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[5]:


x = np.arange(12, 38)
print("Original array:")
print(x)
print("Reverse array:")
x = x[::-1]
print(x)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[6]:


Z = np.zeros(10)
Z[4] = 1
print(Z)


# 13. Create a 3x3 identity matrix

# In[7]:


Z = np.eye(3)
print(Z)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[6]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
print(arr1 * arr2)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[2]:


import numpy as np
arr = np.array([1,2,3,4,5])
print(arr)
print('after converting numpy integer array to float array')
int_array = arr.astype(float)
print(int_array)
print("The data type of int_array is: ")
print(int_array.dtype)


# In[13]:


arr = arr.astype('float')


# In[ ]:





# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[7]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
arr3 = np.intersect1d(arr1,arr2)
print(arr3)


# 17. Extract all odd numbers from arr with values(0-9)

# In[21]:


import numpy as np
arr1 = np.array([0,1,2,3,4,5,6,7,8,9])
arr1[arr1 % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[28]:


import numpy as np
arr1 = np.array([0,1,2,3,4,5,6,7,8,9]
arr1[arr1 > 5 ] = -1
print(arr1)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[2]:


import numpy as np
arr = np.arange(10)
arr


# In[3]:


arr[5:-1] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[5]:


import numpy as np
arr = np.arange(0,9)
arr.reshape(3,3)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[11]:


import numpy as np
arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
bad_vals = [5]
update_vals = [12]
for idx, v in enumerate(bad_vals):
    arr2d[arr2d==v] = update_vals[idx]


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[ ]:





# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[12]:


arr = np.arange(0,9)
arr.reshape(3,3)


# In[14]:


arr[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[15]:


import numpy as np
arr = np.arange(0,9)
arr.reshape(3,3)


# In[24]:


arr[4]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[30]:


arr = np.arange(0,9)
arr.reshape(3,3)


# In[6]:


d2array[0:2,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[8]:


import numpy as np
arr = np.random.randint(100,size=(10,10))
arr


# In[9]:


print(np.min(arr))
print(np.max(arr))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[13]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[14]:


R = np.intersect1d(a,b)
R


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[ ]:





# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[15]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data


# In[17]:


print(data[names != "Will"])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[18]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data


# In[19]:


print(data[names != "Will"])
print(data[names != "Joe"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[26]:


import numpy as np
arr = np.random.randn(1,15)
np.reshape
arr


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[28]:


arr = np.random.randn(1,16).reshape(2,2,4)
arr


# 33. Swap axes of the array you created in Question 32

# In[33]:


data = np.random.randn(1,16).reshape(2,2,4)
data


# In[34]:


data.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[35]:


arr = np.arange(10)
arr = np.sqrt(R)
np.where(R<0.5,0,R)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[36]:


a = np.random.randint(12)
b = np.random.randint(12)
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[37]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[39]:


names = set(names)
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[40]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
print(result)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[42]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray = np.delete(sampleArray, 1, axis= 1)
print(sampleArray)


# In[43]:


newColumn = numpy.array([[10,10,10]])
result = np.column_stack((ini_array, column_to_be_added))
 print ("resultant array", str(result))


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[45]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
A = np.dot(x,y)
A.shape


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[ ]:


arr = np,random(1,2,3,5,6,7,8,4,9,10,11,12,16,15,4,3,13,8,6)

