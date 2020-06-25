```python
from nltk.tokenize import word_tokenize
```


```python
Document=[ "there was a place on my ankle that was itching",
"but I did not scratch it",
"and then my ear began to itch",
"and next my back"]
```


```python
words = set()
for string in Document:
    for j in word_tokenize(string):
        words.add(j)
```


```python
print(words)
```

    {'itch', 'a', 'did', 'it', 'back', 'that', 'then', 'and', 'but', 'my', 'itching', 'began', 'not', 'there', 'ankle', 'place', 'on', 'next', 'was', 'to', 'scratch', 'ear', 'I'}
    


```python
bag_matrix = [[]]
for word in words:
    bag_matrix[0].append(word)
print(bag_matrix)
```

    [['itch', 'a', 'did', 'it', 'back', 'that', 'then', 'and', 'but', 'my', 'itching', 'began', 'not', 'there', 'ankle', 'place', 'on', 'next', 'was', 'to', 'scratch', 'ear', 'I']]
    


```python
for string in Document:
    bag_of_freq=[]
    for label in bag_matrix[0]:
        if label in string:
            count  = string.count(label)
            bag_of_freq.append(count)
        else:
            bag_of_freq.append(0)
    bag_matrix.append(bag_of_freq)
```


```python
print(bag_matrix)
```

    [['itch', 'a', 'did', 'it', 'back', 'that', 'then', 'and', 'but', 'my', 'itching', 'began', 'not', 'there', 'ankle', 'place', 'on', 'next', 'was', 'to', 'scratch', 'ear', 'I'], [1, 6, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [1, 3, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], [0, 2, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    

### Here first row is label whereas other are freq. refer as bag-of-word of fre.


```python
for i in bag_matrix:
    print(i)
```

    ['itch', 'a', 'did', 'it', 'back', 'that', 'then', 'and', 'but', 'my', 'itching', 'began', 'not', 'there', 'ankle', 'place', 'on', 'next', 'was', 'to', 'scratch', 'ear', 'I']
    [1, 6, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0]
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    [1, 3, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    [0, 2, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    


```python

```
