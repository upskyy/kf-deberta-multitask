import argparse
import csv
import logging
import os

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dataset_path", type=str, default="kor-nlu-datasets/KorSTS")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    args = parser.parse_args()

    # Read STSbenchmark dataset and use it as development set
    test_samples = []
    test_file = os.path.join(args.sts_dataset_path, "sts-test.tsv")

    with open(test_file, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")

    model = SentenceTransformer(args.model_name_or_path)
    test_evaluator(model)
