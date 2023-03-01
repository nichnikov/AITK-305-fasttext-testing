import os
import fasttext

model = fasttext.train_unsupervised(os.path.join("data", "all_chats_lem.txt"), model="cbow")
model.save_model(os.path.join("models", "bss_cbow_lem.bin"))

model = fasttext.train_unsupervised(os.path.join("data", "all_chats_lem.txt"),  model='skipgram')
model.save_model(os.path.join("models", "bss_skipgram_lem.bin"))