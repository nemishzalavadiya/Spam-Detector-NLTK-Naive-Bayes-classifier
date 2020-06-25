## Stemming


```python
# import libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
```


```python
text = "Very orderly and methodical he looked, with a hand on each knee, and a loud watch ticking a sonorous sermon under his flapped newly bought waist-coat, as though it pitted its gravity and longevity against the levity and evanescence of the brisk fire."
print(text)
```

    Very orderly and methodical he looked, with a hand on each knee, and a loud watch ticking a sonorous sermon under his flapped newly bought waist-coat, as though it pitted its gravity and longevity against the levity and evanescence of the brisk fire.
    


```python
tokens = word_tokenize(text.lower())
print(tokens)
```

    ['very', 'orderly', 'and', 'methodical', 'he', 'looked', ',', 'with', 'a', 'hand', 'on', 'each', 'knee', ',', 'and', 'a', 'loud', 'watch', 'ticking', 'a', 'sonorous', 'sermon', 'under', 'his', 'flapped', 'newly', 'bought', 'waist-coat', ',', 'as', 'though', 'it', 'pitted', 'its', 'gravity', 'and', 'longevity', 'against', 'the', 'levity', 'and', 'evanescence', 'of', 'the', 'brisk', 'fire', '.']
    


```python
stemmer = PorterStemmer()
porter_stemmed = [stemmer.stem(token) for token in tokens]
print(porter_stemmed)
len(porter_stemmed)
```

    ['veri', 'orderli', 'and', 'method', 'he', 'look', ',', 'with', 'a', 'hand', 'on', 'each', 'knee', ',', 'and', 'a', 'loud', 'watch', 'tick', 'a', 'sonor', 'sermon', 'under', 'hi', 'flap', 'newli', 'bought', 'waist-coat', ',', 'as', 'though', 'it', 'pit', 'it', 'graviti', 'and', 'longev', 'against', 'the', 'leviti', 'and', 'evanesc', 'of', 'the', 'brisk', 'fire', '.']
    




    47




```python
# snowball stemmer
stemmer = SnowballStemmer("english")
snowball_stemmed = [stemmer.stem(token) for token in tokens]
print(snowball_stemmed)
len(snowball_stemmed)
```

    ['veri', 'order', 'and', 'method', 'he', 'look', ',', 'with', 'a', 'hand', 'on', 'each', 'knee', ',', 'and', 'a', 'loud', 'watch', 'tick', 'a', 'sonor', 'sermon', 'under', 'his', 'flap', 'newli', 'bought', 'waist-coat', ',', 'as', 'though', 'it', 'pit', 'it', 'graviti', 'and', 'longev', 'against', 'the', 'leviti', 'and', 'evanesc', 'of', 'the', 'brisk', 'fire', '.']
    




    47




```python
df = pd.DataFrame({'token': tokens, 'porter_stemmed': porter_stemmed, 'snowball_stemmed': snowball_stemmed})
df = df[['token', 'porter_stemmed', 'snowball_stemmed']]
print(df)
```

              token porter_stemmed snowball_stemmed
    0          very           veri             veri
    1       orderly        orderli            order
    2           and            and              and
    3    methodical         method           method
    4            he             he               he
    5        looked           look             look
    6             ,              ,                ,
    7          with           with             with
    8             a              a                a
    9          hand           hand             hand
    10           on             on               on
    11         each           each             each
    12         knee           knee             knee
    13            ,              ,                ,
    14          and            and              and
    15            a              a                a
    16         loud           loud             loud
    17        watch          watch            watch
    18      ticking           tick             tick
    19            a              a                a
    20     sonorous          sonor            sonor
    21       sermon         sermon           sermon
    22        under          under            under
    23          his             hi              his
    24      flapped           flap             flap
    25        newly          newli            newli
    26       bought         bought           bought
    27   waist-coat     waist-coat       waist-coat
    28            ,              ,                ,
    29           as             as               as
    30       though         though           though
    31           it             it               it
    32       pitted            pit              pit
    33          its             it               it
    34      gravity        graviti          graviti
    35          and            and              and
    36    longevity         longev           longev
    37      against        against          against
    38          the            the              the
    39       levity         leviti           leviti
    40          and            and              and
    41  evanescence        evanesc          evanesc
    42           of             of               of
    43          the            the              the
    44        brisk          brisk            brisk
    45         fire           fire             fire
    46            .              .                .
    


```python
df[(df.token != df.porter_stemmed) | (df.token != df.snowball_stemmed)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>token</th>
      <th>porter_stemmed</th>
      <th>snowball_stemmed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>very</td>
      <td>veri</td>
      <td>veri</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orderly</td>
      <td>orderli</td>
      <td>order</td>
    </tr>
    <tr>
      <th>3</th>
      <td>methodical</td>
      <td>method</td>
      <td>method</td>
    </tr>
    <tr>
      <th>5</th>
      <td>looked</td>
      <td>look</td>
      <td>look</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ticking</td>
      <td>tick</td>
      <td>tick</td>
    </tr>
    <tr>
      <th>20</th>
      <td>sonorous</td>
      <td>sonor</td>
      <td>sonor</td>
    </tr>
    <tr>
      <th>23</th>
      <td>his</td>
      <td>hi</td>
      <td>his</td>
    </tr>
    <tr>
      <th>24</th>
      <td>flapped</td>
      <td>flap</td>
      <td>flap</td>
    </tr>
    <tr>
      <th>25</th>
      <td>newly</td>
      <td>newli</td>
      <td>newli</td>
    </tr>
    <tr>
      <th>32</th>
      <td>pitted</td>
      <td>pit</td>
      <td>pit</td>
    </tr>
    <tr>
      <th>33</th>
      <td>its</td>
      <td>it</td>
      <td>it</td>
    </tr>
    <tr>
      <th>34</th>
      <td>gravity</td>
      <td>graviti</td>
      <td>graviti</td>
    </tr>
    <tr>
      <th>36</th>
      <td>longevity</td>
      <td>longev</td>
      <td>longev</td>
    </tr>
    <tr>
      <th>39</th>
      <td>levity</td>
      <td>leviti</td>
      <td>leviti</td>
    </tr>
    <tr>
      <th>41</th>
      <td>evanescence</td>
      <td>evanesc</td>
      <td>evanesc</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
