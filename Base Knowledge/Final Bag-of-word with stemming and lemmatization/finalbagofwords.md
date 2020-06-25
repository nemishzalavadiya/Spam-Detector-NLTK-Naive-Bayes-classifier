### Bag of words model


```python
# load all necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_colwidth', 100)
```

#### Let's build a basic bag of words model on three sample documents


```python
documents = ["Gangs of Wasseypur is a great movie.", "The success of a movie depends on the performance of the actors.", "There are no new movies releasing this week."]
print(documents)
```

    ['Gangs of Wasseypur is a great movie.', 'The success of a movie depends on the performance of the actors.', 'There are no new movies releasing this week.']
    


```python
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

```

    ['gangs wasseypur great movie .', 'success movie depends performance actors .', 'new movies releasing week .']
    

#### Creating bag of words model using count vectorizer function


```python
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(documents)
print(bow_model)  # returns the row number and column number of the cells which have 1 as value
```

      (0, 2)	1
      (0, 10)	1
      (0, 3)	1
      (0, 4)	1
      (1, 4)	1
      (1, 9)	1
      (1, 1)	1
      (1, 7)	1
      (1, 0)	1
      (2, 6)	1
      (2, 5)	1
      (2, 8)	1
      (2, 11)	1
    


```python
# print the full sparse matrix
print(bow_model.toarray())
```

    [[0 0 1 1 1 0 0 0 0 0 1 0]
     [1 1 0 0 1 0 0 1 0 1 0 0]
     [0 0 0 0 0 1 1 0 1 0 0 1]]
    


```python
print(bow_model.shape)
print(vectorizer.get_feature_names())
```

    (3, 12)
    ['actors', 'depends', 'gangs', 'great', 'movie', 'movies', 'new', 'performance', 'releasing', 'success', 'wasseypur', 'week']
    

### Let's create a bag of words model on the spam dataset.


```python
# load data
spam = pd.read_csv("SMSSpamCollection.txt", sep = "\t", names=["label", "message"])
spam.head()
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
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives around here though</td>
    </tr>
  </tbody>
</table>
</div>



##### Let's take a subset of data (first 50 rows only) and create bag of word model on that.


```python
spam = spam.iloc[0:50,:]
print(spam)
```

       label  \
    0    ham   
    1    ham   
    2   spam   
    3    ham   
    4    ham   
    5   spam   
    6    ham   
    7    ham   
    8   spam   
    9   spam   
    10   ham   
    11  spam   
    12  spam   
    13   ham   
    14   ham   
    15  spam   
    16   ham   
    17   ham   
    18   ham   
    19  spam   
    20   ham   
    21   ham   
    22   ham   
    23   ham   
    24   ham   
    25   ham   
    26   ham   
    27   ham   
    28   ham   
    29   ham   
    30   ham   
    31   ham   
    32   ham   
    33   ham   
    34  spam   
    35   ham   
    36   ham   
    37   ham   
    38   ham   
    39   ham   
    40   ham   
    41   ham   
    42  spam   
    43   ham   
    44   ham   
    45   ham   
    46   ham   
    47   ham   
    48   ham   
    49   ham   
    
                                                                                                    message  
    0   Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g...  
    1                                                                         Ok lar... Joking wif u oni...  
    2   Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ...  
    3                                                     U dun say so early hor... U c already then say...  
    4                                         Nah I don't think he goes to usf, he lives around here though  
    5   FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for ...  
    6                         Even my brother is not like to speak with me. They treat me like aids patent.  
    7   As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your call...  
    8   WINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To ...  
    9   Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with came...  
    10  I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried ...  
    11  SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, ...  
    12  URGENT! You have won a 1 week FREE membership in our Â£100,000 Prize Jackpot! Txt the word: CLAI...  
    13  I've been searching for the right words to thank you for this breather. I promise i wont take yo...  
    14                                                                  I HAVE A DATE ON SUNDAY WITH WILL!!  
    15  XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here...  
    16                                                                           Oh k...i'm watching here:)  
    17                    Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.  
    18                                           Fine if thatÂ’s the way u feel. ThatÂ’s the way its gota b  
    19  England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to...  
    20                                                            Is that seriously how you spell his name?  
    21                                                    Iâ€˜m going to try for 2 months ha ha only joking  
    22                                                So Ã¼ pay first lar... Then when is da stock comin...  
    23             Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already?  
    24                                            Ffffffffff. Alright no way I can meet up with you sooner?  
    25  Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worrie...  
    26                                                                       Lol your always so convincing.  
    27  Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's le...  
    28                          I'm back &amp; we're packing the car now, I'll let you know if there's room  
    29                                     Ahhh. Work. I vaguely remember that! What does it feel like? Lol  
    30  Wait that's still not all that clear, were you not sure about me being sarcastic or that that's ...  
    31  Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child a...  
    32                                                                        K tell me anything about you.  
    33                 For fear of fainting with the of all that housework you just did? Quick have a cuppa  
    34  Thanks for your subscription to Ringtone UK your mobile will be charged Â£5/month Please confirm...  
    35  Yup... Ok i go home look at the timings then i msg Ã¼ again... Xuhui going to learn on 2nd may t...  
    36                                                      Oops, I'll let you know when my roommate's done  
    37                                                                         I see the letter B on my car  
    38                                                                          Anything lor... U decide...  
    39  Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anyth...  
    40                   Pls go ahead with watts. I just wanted to be sure. Do have a great weekend. Abiola  
    41  Did I forget to tell you ? I want you , I need you, I crave you ... But most of all ... I love y...  
    42  07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free noki...  
    43                                                                                  WHO ARE YOU SEEING?  
    44                             Great! I hope you like your man well endowed. I am  &lt;#&gt;  inches...  
    45                                                                     No calls..messages..missed calls  
    46                                                        Didn't you get hep b immunisation in nigeria.  
    47                                                                      Fair enough, anything going on?  
    48                                  Yeah hopefully, if tyler can't do it I could maybe ask around a bit  
    49  U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'...  
    


```python
# extract the messages from the dataframe
messages = spam.message
print(messages)
```

    0     Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there g...
    1                                                                           Ok lar... Joking wif u oni...
    2     Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ...
    3                                                       U dun say so early hor... U c already then say...
    4                                           Nah I don't think he goes to usf, he lives around here though
    5     FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for ...
    6                           Even my brother is not like to speak with me. They treat me like aids patent.
    7     As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your call...
    8     WINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To ...
    9     Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with came...
    10    I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried ...
    11    SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, ...
    12    URGENT! You have won a 1 week FREE membership in our Â£100,000 Prize Jackpot! Txt the word: CLAI...
    13    I've been searching for the right words to thank you for this breather. I promise i wont take yo...
    14                                                                    I HAVE A DATE ON SUNDAY WITH WILL!!
    15    XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here...
    16                                                                             Oh k...i'm watching here:)
    17                      Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.
    18                                             Fine if thatÂ’s the way u feel. ThatÂ’s the way its gota b
    19    England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to...
    20                                                              Is that seriously how you spell his name?
    21                                                      Iâ€˜m going to try for 2 months ha ha only joking
    22                                                  So Ã¼ pay first lar... Then when is da stock comin...
    23               Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already?
    24                                              Ffffffffff. Alright no way I can meet up with you sooner?
    25    Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worrie...
    26                                                                         Lol your always so convincing.
    27    Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's le...
    28                            I'm back &amp; we're packing the car now, I'll let you know if there's room
    29                                       Ahhh. Work. I vaguely remember that! What does it feel like? Lol
    30    Wait that's still not all that clear, were you not sure about me being sarcastic or that that's ...
    31    Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child a...
    32                                                                          K tell me anything about you.
    33                   For fear of fainting with the of all that housework you just did? Quick have a cuppa
    34    Thanks for your subscription to Ringtone UK your mobile will be charged Â£5/month Please confirm...
    35    Yup... Ok i go home look at the timings then i msg Ã¼ again... Xuhui going to learn on 2nd may t...
    36                                                        Oops, I'll let you know when my roommate's done
    37                                                                           I see the letter B on my car
    38                                                                            Anything lor... U decide...
    39    Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anyth...
    40                     Pls go ahead with watts. I just wanted to be sure. Do have a great weekend. Abiola
    41    Did I forget to tell you ? I want you , I need you, I crave you ... But most of all ... I love y...
    42    07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free noki...
    43                                                                                    WHO ARE YOU SEEING?
    44                               Great! I hope you like your man well endowed. I am  &lt;#&gt;  inches...
    45                                                                       No calls..messages..missed calls
    46                                                          Didn't you get hep b immunisation in nigeria.
    47                                                                        Fair enough, anything going on?
    48                                    Yeah hopefully, if tyler can't do it I could maybe ask around a bit
    49    U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'...
    Name: message, dtype: object
    


```python
# convert messages into list
messages = [message for message in messages]
print(messages)
```

    ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...', 'Ok lar... Joking wif u oni...', "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", 'U dun say so early hor... U c already then say...', "Nah I don't think he goes to usf, he lives around here though", "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, Â£1.50 to rcv", 'Even my brother is not like to speak with me. They treat me like aids patent.', "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune", 'WINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.', 'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030', "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.", 'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info', 'URGENT! You have won a 1 week FREE membership in our Â£100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18', "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.", 'I HAVE A DATE ON SUNDAY WITH WILL!!', 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL', "Oh k...i'm watching here:)", 'Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.', 'Fine if thatÂ’s the way u feel. ThatÂ’s the way its gota b', 'England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ãº1.20 POBOXox36504W45WQ 16+', 'Is that seriously how you spell his name?', 'Iâ€˜m going to try for 2 months ha ha only joking', 'So Ã¼ pay first lar... Then when is da stock comin...', 'Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already?', 'Ffffffffff. Alright no way I can meet up with you sooner?', "Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worried. He knows I'm sick when I turn down pizza. Lol", 'Lol your always so convincing.', "Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's left over dinner ? Do you feel my Love ?", "I'm back &amp; we're packing the car now, I'll let you know if there's room", 'Ahhh. Work. I vaguely remember that! What does it feel like? Lol', "Wait that's still not all that clear, were you not sure about me being sarcastic or that that's why x doesn't want to live with us", "Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child and he got caught up in that. Till 2! But we won't go there! Not doing too badly cheers. You? ", 'K tell me anything about you.', 'For fear of fainting with the of all that housework you just did? Quick have a cuppa', 'Thanks for your subscription to Ringtone UK your mobile will be charged Â£5/month Please confirm by replying YES or NO. If you reply NO you will not be charged', 'Yup... Ok i go home look at the timings then i msg Ã¼ again... Xuhui going to learn on 2nd may too but her lesson is at 8am', "Oops, I'll let you know when my roommate's done", 'I see the letter B on my car', 'Anything lor... U decide...', "Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomo. Not that i'm trying to invite myself or anything!", 'Pls go ahead with watts. I just wanted to be sure. Do have a great weekend. Abiola', 'Did I forget to tell you ? I want you , I need you, I crave you ... But most of all ... I love you my sweet Arabian steed ... Mmmmmm ... Yummy', '07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow', 'WHO ARE YOU SEEING?', 'Great! I hope you like your man well endowed. I am  &lt;#&gt;  inches...', 'No calls..messages..missed calls', "Didn't you get hep b immunisation in nigeria.", 'Fair enough, anything going on?', "Yeah hopefully, if tyler can't do it I could maybe ask around a bit", "U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'm not a weak sucker. Hospitals are for weak suckers."]
    


```python
# preprocess messages using the preprocess function
messages = [preprocess(message) for message in messages]
print(messages)
```

    ['go jurong point , crazy .. available bugis n great world la e buffet ... cine got amore wat ...', 'ok lar ... joking wif u oni ...', "free entry 2 wkly comp win fa cup final tkts 21st may 2005. text fa 87121 receive entry question ( std txt rate ) & c 's apply 08452810075over18 's", 'u dun say early hor ... u c already say ...', "nah n't think goes usf , lives around though", "freemsg hey darling 's 3 week 's word back ! 'd like fun still ? tb ok ! xxx std chgs send , â£1.50 rcv", 'even brother like speak . treat like aids patent .', "per request 'melle melle ( oru minnaminunginte nurungu vettam ) ' set callertune callers . press * 9 copy friends callertune", 'winner ! ! valued network customer selected receivea â£900 prize reward ! claim call 09061701461. claim code kl341 . valid 12 hours .', 'mobile 11 months ? u r entitled update latest colour mobiles camera free ! call mobile update co free 08002986030', "'m gon na home soon n't want talk stuff anymore tonight , k ? 've cried enough today .", 'six chances win cash ! 100 20,000 pounds txt > csh11 send 87575. cost 150p/day , 6days , 16+ tsandcs apply reply hl 4 info', 'urgent ! 1 week free membership â£100,000 prize jackpot ! txt word : claim : 81010 & c www.dbuk.net lccltd pobox 4403ldnw1a7rw18', "'ve searching right words thank breather . promise wont take help granted fulfil promise . wonderful blessing times .", 'date sunday ! !', 'xxxmobilemovieclub : use credit , click wap link next txt message click > > http : //wap . xxxmobilemovieclub.com ? n=qjkgighjjgcbl', "oh k ... 'm watching : )", 'eh u remember 2 spell name ... yes . v naughty make v wet .', 'fine thatâ ’ way u feel . thatâ ’ way gota b', 'england v macedonia - dont miss goals/team news . txt ur national team 87077 eg england 87077 try : wales , scotland 4txt/ãº1.20 poboxox36504w45wq 16+', 'seriously spell name ?', 'iâ€˜m going try 2 months ha ha joking', 'ã¼ pay first lar ... da stock comin ...', 'aft finish lunch go str lor . ard 3 smth lor . u finish ur lunch already ?', 'ffffffffff . alright way meet sooner ?', "forced eat slice . 'm really hungry tho . sucks . mark getting worried . knows 'm sick turn pizza . lol", 'lol always convincing .', "catch bus ? frying egg ? make tea ? eating mom 's left dinner ? feel love ?", "'m back & amp ; 're packing car , 'll let know 's room", 'ahhh . work . vaguely remember ! feel like ? lol', "wait 's still clear , sure sarcastic 's x n't want live us", "yeah got 2 v apologetic . n fallen actin like spoilt child got caught . till 2 ! wo n't go ! badly cheers . ?", 'k tell anything .', 'fear fainting housework ? quick cuppa', 'thanks subscription ringtone uk mobile charged â£5/month please confirm replying yes . reply charged', 'yup ... ok go home look timings msg ã¼ ... xuhui going learn 2nd may lesson 8am', "oops , 'll let know roommate 's done", 'see letter b car', 'anything lor ... u decide ...', "hello ! 's saturday go ? texting see 'd decided anything tomo . 'm trying invite anything !", 'pls go ahead watts . wanted sure . great weekend . abiola', 'forget tell ? want , need , crave ... ... love sweet arabian steed ... mmmmmm ... yummy', '07732584351 - rodger burns - msg = tried call reply sms free nokia mobile + free camcorder . please call 08000930705 delivery tomorrow', 'seeing ?', 'great ! hope like man well endowed . & lt ; # & gt ; inches ...', 'calls .. messages .. missed calls', "n't get hep b immunisation nigeria .", 'fair enough , anything going ?', "yeah hopefully , tyler ca n't could maybe ask around bit", "u n't know stubborn . n't even want go hospital . kept telling mark 'm weak sucker . hospitals weak suckers ."]
    


```python
# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)
print(bow_model.toarray())
```

    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    


```python
print(bow_model.shape)
print(vectorizer.get_feature_names())
```

    (50, 383)
    ['000', '07732584351', '08000930705', '08002986030', '08452810075over18', '09061701461', '100', '11', '12', '150p', '16', '20', '2005', '21st', '2nd', '4403ldnw1a7rw18', '4txt', '50', '6days', '81010', '87077', '87121', '87575', '8am', '900', 'abiola', 'actin', 'aft', 'ahead', 'ahhh', 'aids', 'already', 'alright', 'always', 'amore', 'amp', 'anymore', 'anything', 'apologetic', 'apply', 'arabian', 'ard', 'around', 'ask', 'available', 'back', 'badly', 'bit', 'blessing', 'breather', 'brother', 'buffet', 'bugis', 'burns', 'bus', 'ca', 'call', 'callers', 'callertune', 'calls', 'camcorder', 'camera', 'car', 'cash', 'catch', 'caught', 'chances', 'charged', 'cheers', 'chgs', 'child', 'cine', 'claim', 'clear', 'click', 'co', 'code', 'colour', 'com', 'comin', 'comp', 'confirm', 'convincing', 'copy', 'cost', 'could', 'crave', 'crazy', 'credit', 'cried', 'csh11', 'cup', 'cuppa', 'customer', 'da', 'darling', 'date', 'day', 'dbuk', 'decide', 'decided', 'delivery', 'dinner', 'done', 'dont', 'dun', 'early', 'eat', 'eating', 'eg', 'egg', 'eh', 'endowed', 'england', 'enough', 'entitled', 'entry', 'even', 'fa', 'fainting', 'fair', 'fallen', 'fear', 'feel', 'ffffffffff', 'final', 'fine', 'finish', 'first', 'forced', 'forget', 'free', 'freemsg', 'friends', 'frying', 'fulfil', 'fun', 'get', 'getting', 'go', 'goals', 'goes', 'going', 'gon', 'got', 'gota', 'granted', 'great', 'gt', 'ha', 'hello', 'help', 'hep', 'hey', 'hl', 'home', 'hope', 'hopefully', 'hor', 'hospital', 'hospitals', 'hours', 'housework', 'http', 'hungry', 'immunisation', 'inches', 'info', 'invite', 'iâ', 'jackpot', 'joking', 'jurong', 'kept', 'kl341', 'know', 'knows', 'la', 'lar', 'latest', 'lccltd', 'learn', 'left', 'lesson', 'let', 'letter', 'like', 'link', 'live', 'lives', 'll', 'lol', 'look', 'lor', 'love', 'lt', 'lunch', 'macedonia', 'make', 'man', 'mark', 'may', 'maybe', 'meet', 'melle', 'membership', 'message', 'messages', 'minnaminunginte', 'miss', 'missed', 'mmmmmm', 'mobile', 'mobiles', 'mom', 'month', 'months', 'msg', 'na', 'nah', 'name', 'national', 'naughty', 'need', 'net', 'network', 'news', 'next', 'nigeria', 'nokia', 'nurungu', 'oh', 'ok', 'oni', 'oops', 'oru', 'packing', 'patent', 'pay', 'per', 'pizza', 'please', 'pls', 'pobox', 'poboxox36504w45wq', 'point', 'pounds', 'press', 'prize', 'promise', 'qjkgighjjgcbl', 'question', 'quick', 'rate', 'rcv', 're', 'really', 'receive', 'receivea', 'remember', 'reply', 'replying', 'request', 'reward', 'right', 'ringtone', 'rodger', 'room', 'roommate', 'sarcastic', 'saturday', 'say', 'scotland', 'searching', 'see', 'seeing', 'selected', 'send', 'seriously', 'set', 'sick', 'six', 'slice', 'sms', 'smth', 'soon', 'sooner', 'speak', 'spell', 'spoilt', 'std', 'steed', 'still', 'stock', 'str', 'stubborn', 'stuff', 'subscription', 'sucker', 'suckers', 'sucks', 'sunday', 'sure', 'sweet', 'take', 'talk', 'tb', 'tea', 'team', 'tell', 'telling', 'text', 'texting', 'thank', 'thanks', 'thatâ', 'think', 'tho', 'though', 'till', 'times', 'timings', 'tkts', 'today', 'tomo', 'tomorrow', 'tonight', 'treat', 'tried', 'try', 'trying', 'tsandcs', 'turn', 'txt', 'tyler', 'uk', 'update', 'ur', 'urgent', 'us', 'use', 'usf', 'vaguely', 'valid', 'valued', 've', 'vettam', 'wait', 'wales', 'want', 'wanted', 'wap', 'wat', 'watching', 'watts', 'way', 'weak', 'week', 'weekend', 'well', 'wet', 'wif', 'win', 'winner', 'wkly', 'wo', 'wonderful', 'wont', 'word', 'words', 'work', 'world', 'worried', 'www', 'xuhui', 'xxx', 'xxxmobilemovieclub', 'yeah', 'yes', 'yummy', 'yup', 'ãº1', 'ã¼']
    

* A lot of duplicate tokens such as 'win'and 'winner'; 'reply' and 'replying'; 'want' and 'wanted' etc. 

## Stemming and lemmatising


```python
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
```

### Bag of words model on stemmed messages


```python
# stem messages
messages = [preprocess(message, stem=True) for message in spam.message]

# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)
```


```python
# look at the dataframe
pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names())
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
      <th>000</th>
      <th>07732584351</th>
      <th>08000930705</th>
      <th>08002986030</th>
      <th>08452810075over18</th>
      <th>09061701461</th>
      <th>100</th>
      <th>11</th>
      <th>12</th>
      <th>150p</th>
      <th>...</th>
      <th>www</th>
      <th>xuhui</th>
      <th>xxx</th>
      <th>xxxmobilemovieclub</th>
      <th>ye</th>
      <th>yeah</th>
      <th>yummi</th>
      <th>yup</th>
      <th>ãº1</th>
      <th>ã¼</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 359 columns</p>
</div>




```python
# token names
print(vectorizer.get_feature_names())
```

    ['000', '07732584351', '08000930705', '08002986030', '08452810075over18', '09061701461', '100', '11', '12', '150p', '16', '20', '2005', '21st', '2nd', '4403ldnw1a7rw18', '4txt', '50', '6day', '81010', '87077', '87121', '87575', '8am', '900', 'abiola', 'actin', 'aft', 'ahead', 'ahhh', 'aid', 'alreadi', 'alright', 'alway', 'amor', 'amp', 'anymor', 'anyth', 'apologet', 'appli', 'arabian', 'ard', 'around', 'ask', 'avail', 'back', 'badli', 'bit', 'bless', 'breather', 'brother', 'bu', 'buffet', 'bugi', 'burn', 'ca', 'call', 'caller', 'callertun', 'camcord', 'camera', 'car', 'cash', 'catch', 'caught', 'chanc', 'charg', 'cheer', 'chg', 'child', 'cine', 'claim', 'clear', 'click', 'co', 'code', 'colour', 'com', 'comin', 'comp', 'confirm', 'convinc', 'copi', 'cost', 'could', 'crave', 'crazi', 'credit', 'cri', 'csh11', 'cup', 'cuppa', 'custom', 'da', 'darl', 'date', 'day', 'dbuk', 'decid', 'deliveri', 'dinner', 'done', 'dont', 'dun', 'earli', 'eat', 'eg', 'egg', 'eh', 'endow', 'england', 'enough', 'entitl', 'entri', 'even', 'fa', 'faint', 'fair', 'fallen', 'fear', 'feel', 'ffffffffff', 'final', 'fine', 'finish', 'first', 'forc', 'forget', 'free', 'freemsg', 'fri', 'friend', 'fulfil', 'fun', 'get', 'go', 'goals', 'goe', 'gon', 'got', 'gota', 'grant', 'great', 'gt', 'ha', 'hello', 'help', 'hep', 'hey', 'hl', 'home', 'hope', 'hor', 'hospit', 'hour', 'housework', 'http', 'hungri', 'immunis', 'inch', 'info', 'invit', 'iâ', 'jackpot', 'joke', 'jurong', 'kept', 'kl341', 'know', 'la', 'lar', 'latest', 'lccltd', 'learn', 'left', 'lesson', 'let', 'letter', 'like', 'link', 'live', 'll', 'lol', 'look', 'lor', 'love', 'lt', 'lunch', 'macedonia', 'make', 'man', 'mark', 'may', 'mayb', 'meet', 'mell', 'membership', 'messag', 'minnaminungint', 'miss', 'mmmmmm', 'mobil', 'mom', 'month', 'msg', 'na', 'nah', 'name', 'nation', 'naughti', 'need', 'net', 'network', 'news', 'next', 'nigeria', 'nokia', 'nurungu', 'oh', 'ok', 'oni', 'oop', 'oru', 'pack', 'patent', 'pay', 'per', 'pizza', 'pl', 'pleas', 'pobox', 'poboxox36504w45wq', 'point', 'pound', 'press', 'prize', 'promis', 'qjkgighjjgcbl', 'question', 'quick', 'rate', 'rcv', 're', 'realli', 'receiv', 'receivea', 'rememb', 'repli', 'request', 'reward', 'right', 'rington', 'rodger', 'room', 'roommat', 'sarcast', 'saturday', 'say', 'scotland', 'search', 'see', 'select', 'send', 'serious', 'set', 'sick', 'six', 'slice', 'sm', 'smth', 'soon', 'sooner', 'speak', 'spell', 'spoilt', 'std', 'steed', 'still', 'stock', 'str', 'stubborn', 'stuff', 'subscript', 'suck', 'sucker', 'sunday', 'sure', 'sweet', 'take', 'talk', 'tb', 'tea', 'team', 'tell', 'text', 'thank', 'thatâ', 'think', 'tho', 'though', 'till', 'time', 'tkt', 'today', 'tomo', 'tomorrow', 'tonight', 'treat', 'tri', 'tsandc', 'turn', 'txt', 'tyler', 'uk', 'updat', 'ur', 'urgent', 'us', 'use', 'usf', 'vagu', 'valid', 'valu', 've', 'vettam', 'wait', 'wale', 'want', 'wap', 'wat', 'watch', 'watt', 'way', 'weak', 'week', 'weekend', 'well', 'wet', 'wif', 'win', 'winner', 'wkli', 'wo', 'wonder', 'wont', 'word', 'work', 'world', 'worri', 'www', 'xuhui', 'xxx', 'xxxmobilemovieclub', 'ye', 'yeah', 'yummi', 'yup', 'ãº1', 'ã¼']
    

### 359 tokens after lemmatizing the messages as compared to 381 tokens without stemming.

### Let's try lemmatizing the messages.


```python
# lemmatise messages
messages = [preprocess(message, stem=False) for message in spam.message]

# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)
```


```python
# look at the dataframe
pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names())
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
      <th>000</th>
      <th>07732584351</th>
      <th>08000930705</th>
      <th>08002986030</th>
      <th>08452810075over18</th>
      <th>09061701461</th>
      <th>100</th>
      <th>11</th>
      <th>12</th>
      <th>150p</th>
      <th>...</th>
      <th>www</th>
      <th>xuhui</th>
      <th>xxx</th>
      <th>xxxmobilemovieclub</th>
      <th>yeah</th>
      <th>yes</th>
      <th>yummy</th>
      <th>yup</th>
      <th>ãº1</th>
      <th>ã¼</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 362 columns</p>
</div>




```python
# token names
print(vectorizer.get_feature_names())
```

    ['000', '07732584351', '08000930705', '08002986030', '08452810075over18', '09061701461', '100', '11', '12', '150p', '16', '20', '2005', '21st', '2nd', '4403ldnw1a7rw18', '4txt', '50', '6days', '81010', '87077', '87121', '87575', '8am', '900', 'abiola', 'actin', 'aft', 'ahead', 'ahhh', 'aid', 'already', 'alright', 'always', 'amore', 'amp', 'anymore', 'anything', 'apologetic', 'apply', 'arabian', 'ard', 'around', 'ask', 'available', 'back', 'badly', 'bite', 'bless', 'breather', 'brother', 'buffet', 'bugis', 'burn', 'bus', 'ca', 'call', 'callers', 'callertune', 'camcorder', 'camera', 'car', 'cash', 'catch', 'chance', 'charge', 'cheer', 'chgs', 'child', 'cine', 'claim', 'clear', 'click', 'co', 'code', 'colour', 'com', 'comin', 'comp', 'confirm', 'convince', 'copy', 'cost', 'could', 'crave', 'crazy', 'credit', 'cry', 'csh11', 'cup', 'cuppa', 'customer', 'da', 'darling', 'date', 'day', 'dbuk', 'decide', 'delivery', 'dinner', 'do', 'dont', 'dun', 'early', 'eat', 'eg', 'egg', 'eh', 'endow', 'england', 'enough', 'entitle', 'entry', 'even', 'fa', 'faint', 'fair', 'fall', 'fear', 'feel', 'ffffffffff', 'final', 'fine', 'finish', 'first', 'force', 'forget', 'free', 'freemsg', 'friends', 'fry', 'fulfil', 'fun', 'get', 'go', 'goals', 'gon', 'gota', 'grant', 'great', 'gt', 'ha', 'hello', 'help', 'hep', 'hey', 'hl', 'home', 'hope', 'hopefully', 'hor', 'hospital', 'hospitals', 'hours', 'housework', 'http', 'hungry', 'immunisation', 'inch', 'info', 'invite', 'iâ', 'jackpot', 'joke', 'jurong', 'keep', 'kl341', 'know', 'la', 'lar', 'latest', 'lccltd', 'learn', 'leave', 'lesson', 'let', 'letter', 'like', 'link', 'live', 'll', 'lol', 'look', 'lor', 'love', 'lt', 'lunch', 'macedonia', 'make', 'man', 'mark', 'may', 'maybe', 'meet', 'melle', 'membership', 'message', 'minnaminunginte', 'miss', 'mmmmmm', 'mobile', 'mobiles', 'mom', 'month', 'months', 'msg', 'na', 'nah', 'name', 'national', 'naughty', 'need', 'net', 'network', 'news', 'next', 'nigeria', 'nokia', 'nurungu', 'oh', 'ok', 'oni', 'oops', 'oru', 'pack', 'patent', 'pay', 'per', 'pizza', 'please', 'pls', 'pobox', 'poboxox36504w45wq', 'point', 'pound', 'press', 'prize', 'promise', 'qjkgighjjgcbl', 'question', 'quick', 'rate', 'rcv', 're', 'really', 'receive', 'receivea', 'remember', 'reply', 'request', 'reward', 'right', 'ringtone', 'rodger', 'room', 'roommate', 'sarcastic', 'saturday', 'say', 'scotland', 'search', 'see', 'select', 'send', 'seriously', 'set', 'sick', 'six', 'slice', 'sms', 'smth', 'soon', 'sooner', 'speak', 'spell', 'spoil', 'std', 'steed', 'still', 'stock', 'str', 'stubborn', 'stuff', 'subscription', 'suck', 'sucker', 'suckers', 'sunday', 'sure', 'sweet', 'take', 'talk', 'tb', 'tea', 'team', 'tell', 'text', 'texting', 'thank', 'thatâ', 'think', 'tho', 'though', 'till', 'time', 'tkts', 'today', 'tomo', 'tomorrow', 'tonight', 'treat', 'try', 'tsandcs', 'turn', 'txt', 'tyler', 'uk', 'update', 'ur', 'urgent', 'us', 'use', 'usf', 'vaguely', 'valid', 'value', 've', 'vettam', 'wait', 'wales', 'want', 'wap', 'wat', 'watch', 'watts', 'way', 'weak', 'week', 'weekend', 'well', 'wet', 'wif', 'win', 'winner', 'wkly', 'wo', 'wonderful', 'wont', 'word', 'work', 'world', 'worry', 'www', 'xuhui', 'xxx', 'xxxmobilemovieclub', 'yeah', 'yes', 'yummy', 'yup', 'ãº1', 'ã¼']
    

### 363 tokens after lemmatizing the messages as compared to 381 tokens without lemmatising. But, on the other hand, stemmer reduces the token count to 359. Lemmatization doesn't work as expected because the data is very unclean.
