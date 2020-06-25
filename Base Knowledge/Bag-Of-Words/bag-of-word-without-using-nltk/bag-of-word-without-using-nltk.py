#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.tokenize import word_tokenize


# In[4]:


Document=[ "there was a place on my ankle that was itching",
"but I did not scratch it",
"and then my ear began to itch",
"and next my back"]


# In[18]:


words = set()
for string in Document:
    for j in word_tokenize(string):
        words.add(j)


# In[19]:


print(words)


# In[25]:


bag_matrix = [[]]
for word in words:
    bag_matrix[0].append(word)
print(bag_matrix)


# In[26]:


for string in Document:
    bag_of_freq=[]
    for label in bag_matrix[0]:
        if label in string:
            count  = string.count(label)
            bag_of_freq.append(count)
        else:
            bag_of_freq.append(0)
    bag_matrix.append(bag_of_freq)


# In[27]:


print(bag_matrix)


# ### Here first row is label whereas other are freq. refer as bag-of-word of fre.

# In[28]:


for i in bag_matrix:
    print(i)


# In[ ]:




