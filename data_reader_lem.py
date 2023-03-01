import os
import re
import pandas as pd
from itertools import chain
from texts_processing import TextsTokenizer
# from utils import chunks

"""
tknz = TextsTokenizer()

#chats_files_names = ["Чаты_Главбух_2020_год.txt", "Чаты_Главбух_2021_год.txt", "Чаты_Главбух_2022_год.txt"]
chats_files_names = ["Чаты_Главбух_2022_год.txt"]

lm_txts_str = ""
for num, fn in enumerate(chats_files_names):
    df = pd.read_csv(os.path.join("data", "texts_for_learning", fn), sep="\|",
                     skipinitialspace=True, error_bad_lines=False, names=["chat_id", "text", "created"])


    lem_texts = tknz([re.sub(r"\s+", " ", tx) for tx in list(df["text"])])
    print([x for x in chain(*lem_texts)][:10])
    lm_txt_str = " ".join([x for x in chain(*lem_texts)])

    file_name = "".join(["chats_lem", str(2), ".txt"])
    with open(os.path.join("data", file_name), "w") as tx_f:
        tx_f.write(lm_txt_str)
"""

file_names = ["chats_lem0.txt", "chats_lem1.txt", "chats_lem2.txt"]

all_chats = ""
for fn in file_names:
    with open(os.path.join("data", fn), "r") as f:
        chats = f.read()
    all_chats += " " + chats

with open(os.path.join("data", "all_chats_lem.txt"), "w") as tx_f:
    tx_f.write(all_chats)

