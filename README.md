# spam-detector-NLTK-Naive-Bayes-classifier
NLTK | Naive Bayes Classifier | Tf-Idf Model / Bag-Of-Word Model | Pre-processing

> To use model I have provided pickle file.
  
  Use pickle library to import and get the file. Then use nltk librarie's function like, instance.classify(document) method to see the result.
  - Result will going to be [ Spam / Ham ]
  
 
> Why Naive Bayes ?
- As we know, In spam detection we uses word which is matter, so here words are independent of each other. it's not depend on previous or next word so it's going to be best.
- Here this model's accuracy :
  - Training Set : 99.87657243 [ here i have used 8K+ messages ]
  - Test Set     : 98.5647832  [ here i have used 4k+ messages ]
