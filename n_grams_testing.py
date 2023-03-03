import os
from src.config import PROJECT_ROOT_DIR
import nltk


with open(os.path.join(PROJECT_ROOT_DIR, "data", "chats_lem1.txt"), "r") as tf:
    texts = tf.read()

tokenized_text = texts.split()
print(tokenized_text[:10])

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(tokenized_text)

finder.apply_freq_filter(100)
top_brms = finder.nbest(bigram_measures.pmi, 1000)

for num, bg in enumerate(top_brms):
    print(num, bg)

"""
from nltk import ngrams
n = 2
sixgrams = ngrams(tokenized_text[:10000], n)

for grams in sixgrams:
  print(grams)"""

