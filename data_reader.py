import os
import re
import pandas as pd

chats_files_names = ["Чаты_Главбух_2020_год.txt", "Чаты_Главбух_2021_год.txt", "Чаты_Главбух_2022_год.txt"]

texts = ""
for fn in chats_files_names:
    df = pd.read_csv(os.path.join("data", "texts_for_learning", fn), sep="\|",
                 skipinitialspace=True, error_bad_lines=False, names=["chat_id", "text", "created"])

    texts += " ".join([re.sub(r"\s+", " ", tx) for tx in list(df["text"])])


with open(os.path.join("data", "all_chats.txt"), "w") as tx_f:
    tx_f.write(texts)
