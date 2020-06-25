#!/usr/bin/env python
# coding: utf-8

# ### TF-IDF model

# In[1]:


# load all necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('max_colwidth', 100)


# #### Let's build a basic bag of words model on three sample documents

# In[2]:


documents = ["Gangs of Wasseypur is a great movie. Wasseypur is a town in Bihar.", "The success of a song depends on the music.", "There is a new movie releasing this week. The movie is fun to watch."]
print(documents)


# In[3]:


documents = ["Vapour, Bangalore has a really great terrace seating and an awesome view of the Bangalore skyline",
             "The beer at Vapour, Bangalore was amazing. My favorites are the wheat beer and the ale beer.",
             "Vapour, Bangalore has the best view in Bangalore."]
print(documents)


# In[4]:


from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# add stemming and lemmatisation in the preprocess function
def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    
    # stem
    #words = [stemmer.stem(word) for word in words]
    
    # join words to make sentence
    document = " ".join(words)
    
    return document


# In[5]:


documents = [preprocess(document) for document in documents]
print(documents)


# #### Creating bag of words model using count vectorizer function

# In[6]:


vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(documents)
print(tfidf_model)  # returns the row number and column number of cells which have 1 as value


# In[7]:


# print the full sparse matrix
print(tfidf_model.toarray())


# In[8]:


pd.DataFrame(tfidf_model.toarray(), columns = vectorizer.get_feature_names())


# ### Let's create a tf-idf model on the spam dataset.

# In[10]:


# load data
spam = pd.read_csv("SMSSpamCollection.txt", sep = "\t", names=["label", "message"])
spam.head()


# ##### Let's take a subset of data (first 50 rows only) and create bag of word model on that.

# In[11]:


spam = spam.iloc[0:50,:]
print(spam)


# In[12]:


# extract the messages from the dataframe
messages = [message for message in spam.message]
print(messages)


# In[13]:


# preprocess messages using the preprocess function
messages = [preprocess(message) for message in messages]
print(messages)


# In[14]:


# bag of words model
vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(messages)


# In[15]:


# Let's look at the dataframe
tfidf = pd.DataFrame(tfidf_model.toarray(), columns = vectorizer.get_feature_names())
tfidf


# In[16]:


# token names
print(vectorizer.get_feature_names())


# In[ ]:




