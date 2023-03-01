import os
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from texts_processing import TextsTokenizer

texts = ['кто платит транспортный налог', 'когда платится транспортный налог',
         'кто платит транспортный налог', 'кто платит ндс']
texts1 = ['код налогового вычета',
          'код налогового вычета ребенка-инвалида', 'комиссионная торговля', 'комиссионная торговля документооборот']
# texts2 = ['кто платит транспортный налог', 'когда платится транспортный налог', 'кто платит транспортный налог', 'кто платит ндс']

model = fasttext.load_model(os.path.join("models", "bss_cbow_lem.bin"))
# model = fasttext.load_model(os.path.join("models", "bss_cbow.bin"))
v1 = model.get_sentence_vector('когда платится транспортный налог')
v2 = model.get_sentence_vector('кто платит транспортный налог')
v3 = model.get_sentence_vector('кто платит ндс')

tokenizer = TextsTokenizer()
results = []
for tx1, tx2 in [(tx1, tx2) for tx1 in texts for tx2 in texts]:
    lm_tx1 = " ".join(tokenizer([tx1])[0])
    lm_tx2 = " ".join(tokenizer([tx2])[0])
    v1 = model.get_sentence_vector(lm_tx1)
    v2 = model.get_sentence_vector(lm_tx2)
    sim = cosine_similarity([v1, v2])
    print("text 1:", tx1, "text 2:", tx2, "score:", sim[0][1])
    results.append((tx1, tx2, sim[0][1]))

print(results)
