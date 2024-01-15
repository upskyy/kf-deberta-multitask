# To start training, you need to download the KorNLUDatasets first.
# git clone https://github.com/kakaobrain/kor-nlu-datasets.git

# train on STS dataset only
# python training_sts.py --model_name_or_path klue/bert-base
# python training_sts.py --model_name_or_path klue/roberta-base
# python training_sts.py --model_name_or_path klue/roberta-small
# python training_sts.py --model_name_or_path klue/roberta-large
python training_sts.py --model_name_or_path kakaobank/kf-deberta-base

# train on both NLI and STS dataset (multi-task)
# python training_multi_task.py --model_name_or_path klue/bert-base
# python training_multi_task.py --model_name_or_path klue/roberta-base
# python training_multi_task.py --model_name_or_path klue/roberta-small
# python training_multi_task.py --model_name_or_path klue/roberta-large
python training_multi_task.py --model_name_or_path kakaobank/kf-deberta-base

# train on NLI dataset only
# python training_nli.py --model_name_or_path klue/bert-base
# python training_nli.py --model_name_or_path klue/roberta-base
# python training_nli.py --model_name_or_path klue/roberta-small
# python training_nli.py --model_name_or_path klue/roberta-large
python training_nli.py --model_name_or_path kakaobank/kf-deberta-base