## Plotting word frequencies


```python
import requests
from nltk import FreqDist
from nltk.corpus import stopwords
import seaborn as sns
%matplotlib inline
```

Using text of 'Alice in Wonderland' ebook from https://www.gutenberg.org/


```python
url = "https://www.gutenberg.org/files/11/11-0.txt"
alice = requests.get(url)
print(alice.text)
```

Defining a function to plot word frequencies


```python
?FreqDist
```


```python
def plot_word_frequency(words, top_n=10):
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    plot = sns.barplot(labels, counts)
    return plot
```

Plot words frequencies present in the gutenberg corpus 


```python
alice_words = alice.text.split()
plot_word_frequency(alice_words, 15)
```

## Stopwords

Import stopwords from nltk


```python
from nltk.corpus import stopwords
```

Look at the list of stopwords


```python
print(stopwords.words('spanis'))
```

Let's remove stopwords from the following piece of text.


```python
sample_text = "the great aim of education is not knowledge but action"
```

Break text into words


```python
sample_words = sample_text.split()
print(sample_words)
```

Remove stopwords


```python
sample_words = [word for word in sample_words if word not in stopwords.words('english')]
print(sample_words)
```

Join words back to sentence


```python
sample_text = " ".join(sample_words)
print(sample_text)
```

## Removing stopwords in the genesis corpus


```python
no_stops = [word for word in alice_words if word not in stopwords.words("english")]
```


```python
plot_word_frequency(no_stops, 10)
```

Some other things that can be done
* Need to change tokens to lower case
* Need to get rid of punctuations


```python

```
