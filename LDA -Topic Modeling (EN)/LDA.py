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

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = cv.fit_transform(npr['Article'])

print(len(cv.get_feature_names()))


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])
    

LDA = LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)

print(len(LDA.components_))
print(LDA.components_)
print(len(LDA.components_) , len(LDA.components_[0]))


single_topic = LDA.components_[0]
single_topic.argsort()

single_topic.argsort()[-10:]

for index in single_topic.argsort()[-10:]:
    print(cv.get_feature_names()[index])
    
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]], '\n')
    
topic_results = LDA.transform(dtm)
topic_results.shape
topic_results[0].round(2)
topic_results.argmax(axis=1)

npr['Topic'] = topic_results.argmax(axis=1)
npr.head(10)
