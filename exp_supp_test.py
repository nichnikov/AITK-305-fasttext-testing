import os
import pandas as pd
import requests

test_df = pd.read_csv(os.path.join("data", "queries_for_test.csv"), sep="\t")
print(test_df)

