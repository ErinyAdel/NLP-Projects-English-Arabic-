# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 08:13:09 2021
@author: Eriny
"""

'''
Followed Steps:
        1. Identify The Mispelled Word
        2. Find Strings That Are N Edit Distance Away From The Mispelled Word
        3. Filter Suggested Candidates To Retain Only The Ones Found In The Vocabulary
        4. Order Filtered Candidates Based On Word Probabilities
        5. Choose The Most Likely Candidate
        
• Mispelled Word: A Word Is Not Found On The Vocabulary Of The Corpus Of Text 
                  That The System Is Working With

• Editting: An Operation On A String To Change It Into Another String.
• Edit Distance: A Count Of The Number Of Operations Performed On A Word To Edit It.

Types Of Operations:
    a) Insert  : Adding A Letter
    b) Delete  : Removing A Letter
    c) Swap    : Swapping 2 Adjacent Letters
    d) Replace : Change A Letter To Another                

• Word Probability: The Count Of How Many Times It Appears (C(wᵢ)) By The Total Number Of Words 
      P(wᵢ)         In Our Vocabulary (V)   
∟ P(wᵢ) = C(wᵢ) / V                    
       
• Minimum Edit Distance: The Least Number Of Edits Needed To Transform One String Into Another
       (MED)        
'''

import re
import string
from collections import Counter
import numpy as np

##############################################################################

def read_corpus(filename):
    with open(filename, "r", encoding='utf-8-sig') as file:
        lines = file.readlines() ## Read The File Line By Line
        words = []
        for line in lines:
          words += re.findall(r'\w+', line.lower()) ## Put Each Word In Its Lowercase.

    return words

def read_corpus_lines(filename):
    lines = []
    file = open(filename, "r", encoding='utf-8-sig')
    for line in file:
        for letter in line.split(' '):
            lines.append(letter.lower())
            print(letter)

    return lines

def get_count(word_list):
    '''
    Input:
        word_list: a set of words representing the corpus. 
    Output:
        word_count_dict: The wordcount dictionary for each word where
        key is the word and value is its frequency.
    '''
    
    word_count_dict = {}  ## Each Word Count
    word_count_dict = Counter(word_list)
    return word_count_dict

def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where 
        key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and 
        the values are the probability that a word will occur. 
             
    𝑃(𝑤ᵢ) = 𝐶(𝑤ᵢ) / M 
                            • 𝐶(𝑤ᵢ)  is the total number of times 𝑤ᵢ appears in the corpus.
                            • 𝑀 is the total number of words in the corpus.
    '''
    m = sum(word_count_dict.values())
    word_probs = {w: word_count_dict[w] / m for w in word_count_dict.keys()}
    
    return word_probs

##############################################################################
##                          Editting Operations 

def _split(word):
    '''
        Splits The Word To 2 Parts. For Using In 4 Editting Operations
    '''
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]

def delete(word):
    return [l + r[1:] for l,r in _split(word) if r]

def swap(word):
    return [l + r[1] + r[0] + r[2:] for l, r in _split(word) if len(r)>1]

def replace(word):
    letters = string.ascii_lowercase
    return [l + c + r[1:] for l, r in _split(word) if r for c in letters]

def insert(word):
    letters = string.ascii_lowercase
    return [l + c + r for l, r in _split(word) for c in letters]

##############################################################################

def _edit1(word):
    '''
        Get All Unique Candidates In A Dictornary
    '''    
    return set(delete(word) + swap(word) + replace(word) + insert(word))

def _edit2(word):
  return set(e2 for e1 in _edit1(word) for e2 in _edit1(e1))

def correct_spelling(word, vocabulary, word_probability):
    '''
      Get The Word Itself If It Is In The Vocabulary, Or Get Suggested Words For This Word 
      That Are Generated From The 4 Operations If They Are Also In The Vocabulary
    '''  
    if word in vocabulary:
        print(f"\n'{word}' is already correctly spelt")
        return 
    
    suggestions = _edit1(word) or _edit2(word) or [word]
    best_guesses = [w for w in suggestions if w in vocabulary]
      
    return [(w, word_probability[w]) for w in best_guesses]
        

def correct_word(word, corrections):
    if corrections:
        print('\nSuggested Words:', corrections)
        probs = np.array([c[1] for c in corrections])
        best_ix = np.argmax(probs) ## Get The Index Of The Best Suggested Word (Higher Probability)
        correct = corrections[best_ix][0]
        print(f"\n'{correct}' is Suggested for '{word}'")
        return correct

##############################################################################

base_words = read_corpus("./arabic-wordlist-1.6.txt")
vocabs = set(base_words) ## Vocabulary (Unique Words)
word_dict_counts = get_count(base_words)
word_prob   = get_probs(word_dict_counts) 
     
lines = read_corpus_lines("./extracted_data.txt")      
print('\n\n',lines)  


file = open("./autocorrected_extracted_data.txt", "a", encoding='utf-8-sig')
for item in lines:
    corrections = correct_spelling(item, vocabs, word_prob)
    correct     = correct_word(item, corrections)
    if correct: ## Not None
        print(":", correct)
        file.write(correct+' ')
    else:
        print(';',item)
        file.write(item+' ')
file.close()
