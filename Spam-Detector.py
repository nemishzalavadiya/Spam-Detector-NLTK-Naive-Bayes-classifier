#!/usr/bin/env python
# coding: utf-8

# ### SPAM Ham Detection

# In[1]:


import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[2]:


## Reading the given dataset
spam = pd.read_csv("SMSSpamCollection.txt", sep = "\t", names=["label", "message"])


# In[3]:


print(spam.head())


# In[4]:


## Converting the read dataset in to a list of tuples, each tuple(row) contianing the message and it's label
data_set = []
for index,row in spam.iterrows():
    data_set.append((row['message'], row['label']))


# In[5]:


print(data_set[:5])


# In[6]:


print(len(data_set))


# ### Preprocessing

# In[7]:


## initialise the inbuilt Stemmer and the Lemmatizer
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


# In[8]:


def preprocess(document, stem=True):
    'changes document to lower case, removes stopwords and lemmatizes/stems the remainder of the sentence'

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


# In[16]:


get_ipython().run_cell_magic('time', '', '## - Performing the preprocessing steps on all messages\nmessages_set = []\nfor (message, label) in data_set:\n    words_filtered = [e.lower() for e in preprocess(message, stem=False).split() if len(e) >= 3]\n    messages_set.append((words_filtered, label))')


# In[17]:


print(messages_set[:5])


# ### Preparing to create features

# In[18]:


## - creating a single list of all words in the entire dataset for feature list creation

def get_words_in_messages(messages):
    all_words = []
    for (message, label) in messages:
      all_words.extend(message)
    return all_words


# In[19]:


## - creating a final feature list using an intuitive FreqDist, to eliminate all the duplicate words
## Note : we can use the Frequency Distribution of the entire dataset to calculate Tf-Idf scores like we did earlier.

def get_word_features(wordlist):

    #print(wordlist[:10])
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


# In[21]:


## - creating the word features for the entire dataset
word_features = get_word_features(get_words_in_messages(messages_set))
print(len(word_features))


# In[27]:


#print(word_features)
# it gives howmany time word occures in a document
#print(list(nltk.FreqDist(get_words_in_messages(messages_set)).values()))


# ### Preparing to create a train and test set

# In[28]:


## - creating slicing index at 80% threshold
sliceIndex = int((len(messages_set)*.8))


# In[29]:


## - shuffle the pack to create a random and unbiased split of the dataset
random.shuffle(messages_set)


# In[30]:


train_messages, test_messages = messages_set[:sliceIndex], messages_set[sliceIndex:]


# In[32]:


print(len(train_messages))
print(len(test_messages))


# ### Preparing to create feature maps for train and test data

# In[33]:


## creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[34]:


get_ipython().run_cell_magic('time', '', '## - creating the feature map of train and test data\n\ntraining_set = nltk.classify.apply_features(extract_features, train_messages)\ntesting_set = nltk.classify.apply_features(extract_features, test_messages)')


# In[35]:


print(training_set[:5])


# In[36]:


print('Training set size : ', len(training_set))
print('Test set size : ', len(testing_set))


# ### Training

# In[37]:


## Training the classifier with NaiveBayes algorithm
spamClassifier = nltk.NaiveBayesClassifier.train(training_set)


# ### Evaluation

# In[38]:


## - Analyzing the accuracy of the test set
print(nltk.classify.accuracy(spamClassifier, training_set))


# In[39]:


## Analyzing the accuracy of the test set
print(nltk.classify.accuracy(spamClassifier, testing_set))


# In[43]:


## Testing a example message with our newly trained classifier
m = 'CONGRATULATIONS!! As a valued account holder you have been selected to receive a £900 prize reward! Valid 12 hours only.'
print('Classification result : ', spamClassifier.classify(extract_features(m.split())))


# In[47]:


## Priting the most informative features in the classifier
print(spamClassifier.show_most_informative_features())


# In[48]:


## storing the classifier on disk for later usage
import pickle
f = open('nb_spam_classifier.pickle', 'wb')
pickle.dump(spamClassifier,f)
print('Classifier stored at ', f.name)
f.close()

