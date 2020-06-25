#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


import re


# In[3]:


url = r"https://cdn.upgrad.com/UpGrad/temp/bab3e784-e601-4911-9000-f1fbc994a62d/SMSSpamCollection.txt"


# In[4]:


data = requests.get(url).text
print(data)


# In[9]:


pattern = re.compile(r"(ham|spam)[\s]+(.*)")


# In[10]:


for i in pattern.findall(data):
    print(i)


# In[13]:


import pandas as pd


# In[14]:


data_dict = []
for i in pattern.findall(data):
    dic = {"label":i[0],"message":i[1]}
    data_dict.append(dic)
point = pd.DataFrame(data_dict)
print(point)


# In[16]:


point.to_csv("spam_dataset.csv")

