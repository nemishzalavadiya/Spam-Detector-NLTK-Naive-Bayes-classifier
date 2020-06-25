#!/usr/bin/env python
# coding: utf-8

# ### Bag of words model

# In[1]:


# load all necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_colwidth', 100)


# #### Let's build a basic bag of words model on three sample documents

# In[2]:


documents = ["Gangs of Wasseypur is a great movie.", "The success of a movie depends on the performance of the actors.", "There are no new movies releasing this week."]
print(documents)


# In[3]:


def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # join words to make sentence
    document = " ".join(words)
    
    return document

documents = [preprocess(document) for document in documents]
print(documents)


# #### Creating bag of words model using count vectorizer function

# In[4]:


vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(documents)
print(bow_model)  # returns the row number and column number of the cells which have 1 as value


# In[5]:


# print the full sparse matrix
print(bow_model.toarray())


# In[6]:


print(bow_model.shape)
print(vectorizer.get_feature_names())


# ### Let's create a bag of words model on the spam dataset.

# In[8]:


# load data
spam = pd.read_csv("SMSSpamCollection.txt", sep = "\t", names=["label", "message"])
spam.head()


# ##### Let's take a subset of data (first 50 rows only) and create bag of word model on that.

# In[9]:


spam = spam.iloc[0:50,:]
print(spam)


# In[10]:


# extract the messages from the dataframe
messages = spam.message
print(messages)


# In[11]:


# convert messages into list
messages = [message for message in messages]
print(messages)


# In[12]:


# preprocess messages using the preprocess function
messages = [preprocess(message) for message in messages]
print(messages)


# In[13]:


# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)
print(bow_model.toarray())


# In[14]:


print(bow_model.shape)
print(vectorizer.get_feature_names())


# * A lot of duplicate tokens such as 'win'and 'winner'; 'reply' and 'replying'; 'want' and 'wanted' etc. 

# ## Stemming and lemmatising

# In[15]:


from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

# add stemming and lemmatisation in the preprocess function
def preprocess(document, stem=True):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    
    if stem:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    # join words to make sentence
    document = " ".join(words)
    
    return document


# ### Bag of words model on stemmed messages

# In[16]:


# stem messages
messages = [preprocess(message, stem=True) for message in spam.message]

# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)


# In[17]:


# look at the dataframe
pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names())


# In[18]:


# token names
print(vectorizer.get_feature_names())


# ### 359 tokens after lemmatizing the messages as compared to 381 tokens without stemming.
# 
# ### Let's try lemmatizing the messages.

# In[19]:


# lemmatise messages
messages = [preprocess(message, stem=False) for message in spam.message]

# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)


# In[20]:


# look at the dataframe
pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names())


# In[21]:


# token names
print(vectorizer.get_feature_names())


# ### 363 tokens after lemmatizing the messages as compared to 381 tokens without lemmatising. But, on the other hand, stemmer reduces the token count to 359. Lemmatization doesn't work as expected because the data is very unclean.
