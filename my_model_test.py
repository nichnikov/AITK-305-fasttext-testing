import os
import fasttext
import numpy as np


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def div_norm(x):
    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


model = fasttext.load_model(os.path.join("models", "bss_cbow.bin"))
print(model.get_nearest_neighbors('ндфл', k=50))
print(model.get_nearest_neighbors('ндс', k=50))
print(model.get_nearest_neighbors('земельный', k=50))
print(model.get_nearest_neighbors('транспортный', k=50))
print(model.get_nearest_neighbors('транспортный налог', k=50))

one = model.get_word_vector('транспортный')
two = model.get_word_vector('налог')
eos = model.get_word_vector('\n')

# Getting the sentence vector for the sentence "one two" in Finnish.
one_two = model.get_sentence_vector('транспортный налог')
one_three_avg = (div_norm(one) + div_norm(two) + div_norm(eos)) / 3
one_two_avg = (div_norm(one) + div_norm(two)) / 2

print(one_two)
print(one_two_avg)
print(one_three_avg)
