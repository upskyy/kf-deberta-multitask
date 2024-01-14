# kf-deberta-multitask

kakaobank의 [kf-deberta-base](https://huggingface.co/kakaobank/kf-deberta-base) 모델을 KorNLI, KorSTS 데이터셋으로 파인튜닝한 모델입니다.
[jhgan00/ko-sentence-transformers](https://github.com/jhgan00/ko-sentence-transformers) 코드를 기반으로 일부 수정하여 진행하였습니다.

<br>

## KorSTS Benchmark

- [jhgan00/ko-sentence-transformers](https://github.com/jhgan00/ko-sentence-transformers#korsts-benchmarks)의 결과를 참고하여 재작성하였습니다.
- 학습 및 성능 평가 과정은 `training_*.py`, `benchmark.py` 에서 확인할 수 있습니다.
- 학습된 모델은 허깅페이스 모델 허브에 공개되어 있습니다.

<br>

|model|cosine_pearson|cosine_spearman|euclidean_pearson|euclidean_spearman|manhattan_pearson|manhattan_spearman|dot_pearson|dot_spearman|
|:-------------------------|-----------------:|------------------:|--------------------:|---------------------:|--------------------:|---------------------:|--------------:|---------------:|
|[kf-deberta-multitask](https://huggingface.co/upskyy/kf-deberta-multitask)|**85.75**|**86.25**|**84.79**|**85.25**|**84.80**|**85.27**|**82.93**|**82.86**|
|[ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)|84.77|85.6|83.71|84.40|83.70|84.38|82.42|82.33|
|[ko-sbert-multitask](https://huggingface.co/jhgan/ko-sbert-multitask)|84.13|84.71|82.42|82.66|82.41|82.69|80.05|79.69|
|[ko-sroberta-base-nli](https://huggingface.co/jhgan/ko-sroberta-nli)|82.83|83.85|82.87|83.29|82.88|83.28|80.34|79.69|
|[ko-sbert-nli](https://huggingface.co/jhgan/ko-sbert-multitask)|82.24|83.16|82.19|82.31|82.18|82.3|79.3|78.78|
|[ko-sroberta-sts](https://huggingface.co/jhgan/ko-sroberta-sts)|81.84|81.82|81.15|81.25|81.14|81.25|79.09|78.54|
|[ko-sbert-sts](https://huggingface.co/jhgan/ko-sbert-sts)|81.55|81.23|79.94|79.79|79.9|79.75|76.02|75.31|

<br>

## Examples

아래는 임베딩 벡터를 통해 가장 유사한 문장을 찾는 예시입니다.
더 많은 예시는 [sentence-transformers 문서](https://www.sbert.net/index.html)를 참고해주세요.

```python
from transformers import AutoTokenizer, AutoModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ["경제 전문가가 금리 인하에 대한 예측을 하고 있다.", "주식 시장에서 한 투자자가 주식을 매수한다."]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Sentence transformer model for financial domain
embedder = SentenceTransformer("upskyy/kf-deberta-multitask")

# Financial domain corpus
corpus = [
    "주식 시장에서 한 투자자가 주식을 매수한다.",
    "은행에서 예금을 만기로 인출하는 고객이 있다.",
    "금융 전문가가 새로운 투자 전략을 개발하고 있다.",
    "증권사에서 주식 포트폴리오를 관리하는 팀이 있다.",
    "금융 거래소에서 새로운 디지털 자산이 상장된다.",
    "투자 은행가가 고객에게 재무 계획을 제안하고 있다.",
    "금융 회사에서 신용평가 모델을 업데이트하고 있다.",
    "투자자들이 새로운 ICO에 참여하려고 하고 있다.",
    "경제 전문가가 금리 인상에 대한 예측을 내리고 있다.",
]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Financial domain queries
queries = [
    "한 투자자가 비트코인을 매수한다.",
    "은행에서 대출을 상환하는 고객이 있다.",
    "금융 분야에서 새로운 기술 동향을 조사하고 있다."
]

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    # We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in the financial corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
```

<br>

```
======================


Query: 한 투자자가 비트코인을 매수한다.

Top 5 most similar sentences in the financial corpus:
주식 시장에서 한 투자자가 주식을 매수한다. (Score: 0.7579)
투자자들이 새로운 ICO에 참여하려고 하고 있다. (Score: 0.4809)
금융 거래소에서 새로운 디지털 자산이 상장된다. (Score: 0.4669)
금융 전문가가 새로운 투자 전략을 개발하고 있다. (Score: 0.3499)
투자 은행가가 고객에게 재무 계획을 제안하고 있다. (Score: 0.3279)


======================


Query: 은행에서 대출을 상환하는 고객이 있다.

Top 5 most similar sentences in the financial corpus:
은행에서 예금을 만기로 인출하는 고객이 있다. (Score: 0.7762)
금융 회사에서 신용평가 모델을 업데이트하고 있다. (Score: 0.3431)
투자 은행가가 고객에게 재무 계획을 제안하고 있다. (Score: 0.3422)
주식 시장에서 한 투자자가 주식을 매수한다. (Score: 0.2330)
금융 거래소에서 새로운 디지털 자산이 상장된다. (Score: 0.1982)


======================


Query: 금융 분야에서 새로운 기술 동향을 조사하고 있다.

Top 5 most similar sentences in the financial corpus:
금융 거래소에서 새로운 디지털 자산이 상장된다. (Score: 0.5661)
금융 회사에서 신용평가 모델을 업데이트하고 있다. (Score: 0.5184)
금융 전문가가 새로운 투자 전략을 개발하고 있다. (Score: 0.5122)
투자자들이 새로운 ICO에 참여하려고 하고 있다. (Score: 0.4111)
투자 은행가가 고객에게 재무 계획을 제안하고 있다. (Score: 0.3708)
```

<br>

## Training

직접 모델을 파인튜닝하려면 [`kor-nlu-datasets`](https://github.com/kakaobrain/kor-nlu-datasets) 저장소를 clone 하고 `training_*.py` 스크립트를 실행시키면 됩니다.

`train.sh` 파일에서 학습 예시를 확인할 수 있습니다.

```bash
git clone https://github.com/upskyy/kf-deberta-multitask.git
cd kf-deberta-multitask

pip install -r requirements.txt

git clone https://github.com/kakaobrain/kor-nlu-datasets.git

python training_multi_task.py --model_name_or_path kakaobank/kf-deberta-base
./bin/train.sh
```

<br>

## Evaluation

KorSTS Benchmark를 평가하는 방법입니다.

```bash
git clone https://github.com/upskyy/kf-deberta-multitask.git
cd kf-deberta-multitask

pip install -r requirements.txt

git clone https://github.com/kakaobrain/kor-nlu-datasets.git
python bin/benchmark.py
```

<br>

## Export ONNX

`requirements.txt` 설치 후 `bin` 디렉토리에서 `export_onnx.py` 스크립트를 실행시키면 됩니다.

```bash
git clone https://github.com/upskyy/kf-deberta-multitask.git
cd kf-deberta-multitask

pip install -r requirements.txt

python bin/export_onnx.py
```

<br>

## Acknowledgements

- [kakaobank/kf-deberta-base](https://huggingface.co/kakaobank/kf-deberta-base) for pretrained model
- [jhgan00/ko-sentence-transformers](https://github.com/jhgan00/ko-sentence-transformers) for original codebase
- [kor-nlu-datasets](https://github.com/kakaobrain/kor-nlu-datasets) for training data

<br>

## Citation

```bibtex
@proceedings{jeon-etal-2023-kfdeberta,
  title         = {KF-DeBERTa: Financial Domain-specific Pre-trained Language Model},
  author        = {Eunkwang Jeon, Jungdae Kim, Minsang Song, and Joohyun Ryu},
  booktitle     = {Proceedings of the 35th Annual Conference on Human and Cognitive Language Technology},
  moth          = {oct},
  year          = {2023},
  publisher     = {Korean Institute of Information Scientists and Engineers},
  url           = {http://www.hclt.kr/symp/?lnb=conference},
  pages         = {143--148},
}
```

```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
