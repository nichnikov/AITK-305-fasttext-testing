"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
from src.data_types import Parameters
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.utils import timeout, jaccard_similarity
from src.config import logger

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(20)  # timeout


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, tokenizer: TextsTokenizer, parameters: Parameters):
        self.es = ElasticClient()
        self.tkz = tokenizer
        self.prm = parameters

    @timeout(float(tmt))
    def searching(self, text: str, pubid: int, score: float):
        """"""
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                etalons_search_result = self.es.texts_search(self.prm.clusters_index, "LemCluster", [tokens_str])
                print(etalons_search_result)
                if etalons_search_result[0]["search_results"]:
                    for d in etalons_search_result[0]["search_results"]:
                        if pubid in d["ParentPubList"] and jaccard_similarity(tokens_str, d["LemCluster"]) >= score:
                            answers_search_result = self.es.answer_search(self.prm.answers_index, d["ID"], pubid)
                            if answers_search_result["search_results"]:
                                search_result = {"templateId": answers_search_result["search_results"][0]["templateId"],
                                                 "templateText": answers_search_result["search_results"][0][
                                                     "templateText"]}
                                logger.info("search completed successfully with result: {}".format(str(search_result)))
                                return search_result
                            else:
                                logger.info("not found answer with templateId {} and pub_id {}".format(str(d["ID"]),
                                                                                                       str(pubid)))
                        else:
                            logger.info("pubId {} is not in PubsList or (and) low score for "
                                        "text: {}".format(str(pubid), str(tokens_str)))
                    return {"templateId": 0, "templateText": ""}
                else:
                    logger.info("es didn't find anything for text of tokens {}".format(str(tokens_str)))
                    return {"templateId": 0, "templateText": ""}
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return {"templateId": 0, "templateText": ""}
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return {"templateId": 0, "templateText": ""}


if __name__ == "__main__":
    import os
    import time
    import pandas as pd
    from src.config import PROJECT_ROOT_DIR

    t = time.time()
    tknzr = TextsTokenizer()
    stopwords = []
    stopwords_roots = [os.path.join(PROJECT_ROOT_DIR, "data", "greetings.csv"),
                       os.path.join(PROJECT_ROOT_DIR, "data", "stopwords.csv")]

    for root in stopwords_roots:
        stopwords_df = pd.read_csv(root, sep="\t")
        stopwords += list(stopwords_df["text"])
    tknzr.add_stopwords(stopwords)
    print("TextsTokenizer upload:", time.time() - t)

    t0 = time.time()
    c = FastAnswerClassifier(tknzr)
    print("FastAnswerClassifier upload:", time.time() - t0)

    t1 = time.time()
    r = c.searching("как вернули госпошлины по решение судов", 6, 0.95)
    print("searching time:", time.time() - t1)
    print(r)

    t2 = time.time()
    r = c.searching("электрическая электростанция, чебуркша", 6, 0.95)
    print("searching time:", time.time() - t2)
    print(r)

    print("all working time:", time.time() - t)
