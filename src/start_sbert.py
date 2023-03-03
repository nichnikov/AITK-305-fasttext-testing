from src.texts_processing import TextsTokenizer
from src.config import (stopwords,
                        parameters,
                        logger)
from src.classifiers_sbert import FastAnswerClassifier
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, sbert_model)
logger.info("service started...")
