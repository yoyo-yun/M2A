import os

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
if not os.path.exists(PRE_TRAINED_VECTOR_PATH):
    os.makedirs(PRE_TRAINED_VECTOR_PATH)

DATASET_PATH = 'corpus'
CKPTS_PATH = 'ckpts'
LOG_PATH = 'logs'
Global_Huggingface = None

DATASET_PATH_MAP = {
    "imdb": os.path.join(DATASET_PATH, 'imdb'),
    "yelp_13": os.path.join(DATASET_PATH, 'yelp_13'),
    "yelp_14": os.path.join(DATASET_PATH, 'yelp_14'),
    "mtl": os.path.join(DATASET_PATH, "MTL"),
}

if Global_Huggingface is not None:
    MODEL_MAP = {
        'bert': Global_Huggingface + 'bert/' + 'bert-base-uncased',
        'uni_bert': Global_Huggingface + 'bert/' + 'bert-base-uncased',
        'bert_large': 'bert-large-uncased',
        'roberta': Global_Huggingface + 'roberta/' + 'roberta-base',
        'roberta_large': 'roberta-large',
        'flan_t5': 'google/flan-t5-base',
        'user': 'bert-base-uncased',
        "ma_bert": 'bert-base-uncased',
        "gpt2": Global_Huggingface + 'gpt2/'+'gpt2',
    }
else:
    MODEL_MAP = {
        'bert': 'bert-base-uncased',
        'bert_large': 'bert-large-uncased',
        'roberta': 'roberta-base',
        'roberta_large': 'roberta-large',
        'flan_t5': 'google/flan-t5-base',
        'user': 'bert-base-uncased',
        "ma_bert": 'bert-base-uncased',
        'uni_bert': 'bert-base-uncased',
        "gpt2": 'gpt2',
    }