#!/usr/bin/env python
# coding: utf-8

# ## Plotting word frequencies

# In[1]:


import requests
from nltk import FreqDist
from nltk.corpus import stopwords
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Using text of 'Alice in Wonderland' ebook from https://www.gutenberg.org/

# In[2]:


url = "https://www.gutenberg.org/files/11/11-0.txt"
alice = requests.get(url)
print(alice.text)


# Defining a function to plot word frequencies

# In[3]:


get_ipython().run_line_magic('pinfo', 'FreqDist')


# In[4]:


def plot_word_frequency(words, top_n=10):
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    plot = sns.barplot(labels, counts)
    return plot


# Plot words frequencies present in the gutenberg corpus 

# In[5]:


alice_words = alice.text.split()
plot_word_frequency(alice_words, 15)


# ## Stopwords

# Import stopwords from nltk

# In[6]:


from nltk.corpus import stopwords


# Look at the list of stopwords

# In[13]:


print(stopwords.words('spanis'))


# Let's remove stopwords from the following piece of text.

# In[14]:


sample_text = "the great aim of education is not knowledge but action"


# Break text into words

# In[15]:


sample_words = sample_text.split()
print(sample_words)


# Remove stopwords

# In[16]:


sample_words = [word for word in sample_words if word not in stopwords.words('english')]
print(sample_words)


# Join words back to sentence

# In[17]:


sample_text = " ".join(sample_words)
print(sample_text)


# ## Removing stopwords in the genesis corpus

# In[18]:


no_stops = [word for word in alice_words if word not in stopwords.words("english")]


# In[19]:


plot_word_frequency(no_stops, 10)


# Some other things that can be done
# * Need to change tokens to lower case
# * Need to get rid of punctuations
