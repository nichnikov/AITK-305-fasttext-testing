# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# sentences = ["This is an example sentence", "Each sentence is converted"]
# sentences = ['код налогового вычета',
#             'код налогового вычета ребенка-инвалида', 'комиссионная торговля', 'комиссионная торговля документооборот']

sentences = ['если декларация по налогу на имущество за 2022 сдана раньше срока, надо ли предоставлять уведомление об исчисленных налогах',
             'в какой срок сдать декларацию по налогу на имущество за 2022 г., если мы не подаем уведомление о начисленных налогах, а платим платежными поручениями по кбк']
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)

sim = cosine_similarity(embeddings)
print(sim)
