# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:00:05 2021
@author: Eriny
"""

import re
import string
from collections import Counter

class SpellChecker(object):

  def __init__(self, corpus_file_path):
    with open(corpus_file_path, "r") as file:
      lines = file.readlines()
      words = []
      for line in lines:
        words += re.findall(r'\w+', line.lower())

    self.vocabs = set(words)
    self.word_counts = Counter(words)
    total_words = float(sum(self.word_counts.values()))
    self.word_probas = {word: self.word_counts[word] / total_words for word in self.vocabs}

  def _level_one_edits(self, word):
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l,r in splits if r]
    swaps = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r)>1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in letters]
    inserts = [l + c + r for l, r in splits for c in letters] 

    return set(deletes + swaps + replaces + inserts)

  def _level_two_edits(self, word):
    return set(e2 for e1 in self._level_one_edits(word) for e2 in self._level_one_edits(e1))

  def check(self, word):
    candidates = self._level_one_edits(word) or self._level_two_edits(word) or [word]
    valid_candidates = [w for w in candidates if w in self.vocabs]
    return sorted([(c, self.word_probas[c]) for c in valid_candidates], key=lambda tup: tup[1], reverse=True)
    

checker = SpellChecker("./en-vocabulary.txt")

mylist = []

file = open("./extracted_data.txt", "r")
for line in file:
    for letter in line.split(' '):
        mylist.append(letter)
        print(letter)
        
print('\n\n',mylist)  

i = checker.check('badr')
print(i)

file = open("./output.txt", "a")
for item in mylist:
    if item == '\n':
        print("NEW LINEE")
        file.write('\n')
    else:
        i = checker.check(item)
        if i: ## Not None
            print(":", i[0][0])
            file.write(i[0][0]+' ')
        else:
            print(';',item)
            file.write(item+' ')
file.close()

