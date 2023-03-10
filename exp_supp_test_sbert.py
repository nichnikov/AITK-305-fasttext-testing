import os
import pandas as pd
from src.start_sbert import classifier

test_qrs_df = pd.read_csv(os.path.join("data", "queries_for_test.csv"), sep="\t")
test_qrs_df = test_qrs_df


results = []
k = 1
for chat_id, query in zip(test_qrs_df["chat_id"], test_qrs_df["text"]):
    result = classifier.searching(query)
    result_ = (chat_id, query) + result
    results.append(result_)
    print(k, "/", len(test_qrs_df))
    k += 1


print(results)
results_df = pd.DataFrame(results, columns=["ChatId", "Query", "FastAnswID", "Etalon", "LemEtalon", "FTScore", "JacScore"])
print(results_df)
results_df.to_csv(os.path.join("data", "searching_results_sbert.csv"), sep="\t", index=False)