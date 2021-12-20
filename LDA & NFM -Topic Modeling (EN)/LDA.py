# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:08:49 2021
@author: Eriny
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.decomposition import LatentDirichletAllocation

npr = pd.read_csv('npr.csv')
print(npr.head())

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')  ## max_df: Maximum Document Frequency
dtm = cv.fit_transform(npr['Article'])

LDA = LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)

## Number Of (Unique) Words In All Articles.
print(len(cv.get_feature_names()))

print("-- Random 10 Words --")
for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])
    

print(len(LDA.components_), '\n')    ## No. of Components
print(LDA.components_, '\n')         ## The Components
print(len(LDA.components_), '\n')    ## Len of First Component
print(len(LDA.components_[0]), '\n') ## First Component 


single_topic = LDA.components_[0]
single_topic.argsort()

single_topic.argsort()[-10:]
    
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]], '\n')
    
topic_results = LDA.transform(dtm)
topic_results.shape
topic_results[0].round(2)

npr['Topic'] = topic_results.argmax(axis=1)
print(npr.head(10))
