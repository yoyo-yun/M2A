import os
import torch
from M2A.dataset import *
from M2A.cfgs.constants import DATASET_PATH_MAP, MODEL_MAP
from torch.utils.data import DataLoader

DATASET_MAP_BERT = {
    "imdb": data_processor.IMDB,
    "yelp_13": data_processor.YELP_13,
    "yelp_14": data_processor.YELP_14,
    "mtl": data_processor.MTL
}

class Metrics:
    def __init__(self, metrics: [list]):
        self.metrics = metrics

    def add_batch(self, predictions=None, references=None, **kwargs):
        for metric in self.metrics:
            metric.add_batch(
                predictions=predictions,
                references=references,
                **kwargs)

    def merge_results(self, metrics):
        results = {}
        for metric in metrics:
            results.update(metric)
        return results

    def compute(self, **kwargs):
        # return self.merge_results([metric.compute(**kwargs) for metric in self.metrics])
        results = []
        for metric in self.metrics:
            if metric.name in ['f1']:
                results.append(metric.compute(**kwargs))
            else:
                results.append(metric.compute())
        return self.merge_results(results)


class MyVector:
    def __init__(self):
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None


def load_baselines_datasets(path, field='train', strategy='tail'):
    return torch.load(os.path.join(path, '{}_{}.pt'.format(field, strategy)))


class Data(torch.utils.data.Dataset):
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)


def load_vocab(path, field='usr'):
    # itos, stoi, vectors, dim
    return torch.load(os.path.join(path, '{}.pt'.format(field)))


def load_attr_vocab(dataset, users, products):
    try:
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='prd')
    except:
        usr_vocab = build_vocab(users)
        prd_vocab = build_vocab(products)
        save_vectors(DATASET_PATH_MAP[dataset], usr_vocab, field='usr')
        save_vectors(DATASET_PATH_MAP[dataset], prd_vocab, field='prd')
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='prd')
    return usr_stoi, prd_stoi



def save_vectors(path, vocab, field='usr'):
    # itos, stoi, vectors, dim
    data = vocab.get_itos(), vocab.get_stoi()
    torch.save(data, os.path.join(path, '{}.pt'.format(field)))


def build_vocab(counter):
    from torchtext.vocab import vocab
    from collections import OrderedDict
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab = vocab(ordered_dict)
    return vocab


# utils for plms
def saving_dataset_whole_document_plms(config):
    from transformers import AutoTokenizer
    pretrained_weights = MODEL_MAP[config.model]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    processor = DATASET_MAP_BERT[config.dataset]()
    train_examples, dev_examples, test_examples, unlabel_examples = processor.get_documents()

    train_data, dev_data, test_data, unlabel_data = [], [], [], []

    def over_one_example(text, tokenizer):
        tokens = tokenizer.tokenize(text)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        return input_id

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        train_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        dev_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        test_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))
    print("==loading unlabel datasets")
    for step, example in enumerate(unlabel_examples):
        unlabel_data.append(
            ((over_one_example(example.text, tokenizer), example.label, example.user, example.product)))
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                          step / len(test_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    torch.save(train_data, os.path.join(DATASET_PATH_MAP[config.dataset], f'train_whole_{config.model}.pt'))
    torch.save(dev_data, os.path.join(DATASET_PATH_MAP[config.dataset], f'dev_whole_{config.model}.pt'))
    torch.save(test_data, os.path.join(DATASET_PATH_MAP[config.dataset], f'test_whole_{config.model}.pt'))
    torch.save(unlabel_data, os.path.join(DATASET_PATH_MAP[config.dataset], f'unlabel_whole_{config.model}.pt'))

    users, products = processor.get_attributes()
    usr_stoi, prd_stoi = load_attr_vocab(config.dataset, users, products)
    config.num_labels = processor.NUM_CLASSES
    config.num_usrs = len(usr_stoi)
    config.num_prds = len(prd_stoi)


def load_dataset_whole_document_plms(config, from_scratch=True):
    processor = DATASET_MAP_BERT[config.dataset]()
    config.num_labels = processor.NUM_CLASSES
    try:
        if from_scratch: raise Exception
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'train_whole_{config.model}.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'dev_whole_{config.model}.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'test_whole_{config.model}.pt'))
        unlabel_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'unlabel_whole_{config.model}.pt'))
        print(f"===loading document from local for {config.model}...")
        print("Done!")
    except:
        saving_dataset_whole_document_plms(config)
        train_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'train_whole_{config.model}.pt'))
        dev_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'dev_whole_{config.model}.pt'))
        test_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'test_whole_{config.model}.pt'))
        unlabel_data = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], f'unlabel_whole_{config.model}.pt'))

    from M2A.trainer.bucket_iteractor import BucketIteratorForPLMTokenizer
    usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
    prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
    config.usr_size = len(usr_stoi)
    config.prd_size = len(prd_stoi)

    train_dataloader = BucketIteratorForPLMTokenizer(train_data, config.TRAIN.batch_size, config.model, shuffle=True,
                                                     device=config.device, trunc=config.strategy, description="Train",
                                                     stoi=usr_stoi, stoi_p=prd_stoi, max_length=config.max_length)
    dev_dataloader = BucketIteratorForPLMTokenizer(dev_data, config.TRAIN.batch_size, config.model, shuffle=True,
                                                   device=config.device, trunc=config.strategy, description="Dev",
                                                   stoi=usr_stoi, stoi_p=prd_stoi, max_length=config.max_length)
    test_dataloader = BucketIteratorForPLMTokenizer(test_data, config.TRAIN.batch_size, config.model, shuffle=False,
                                                    device=config.device, trunc=config.strategy, description="Test",
                                                    stoi=usr_stoi, stoi_p=prd_stoi, max_length=config.max_length)
    un_dataloader = BucketIteratorForPLMTokenizer(unlabel_data, config.TRAIN.batch_size, config.model, shuffle=False,
                                                    device=config.device, trunc=config.strategy, description="Test",
                                                    stoi=usr_stoi, stoi_p=prd_stoi, max_length=config.max_length)

    config.num_train_optimization_steps = int(
        len(train_dataloader) / config.gradient_accumulation_steps) * config.TRAIN.max_epoch

    return train_dataloader, dev_dataloader, test_dataloader, un_dataloader, usr_stoi, prd_stoi


# parameters in statistics
def print_trainable_parameters(model, verbose=True):
    trainable_params = 0
    all_param = 0
    for n, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            if verbose: print(n)
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def set_trainable(model):
    key_list = ['user_embedding', 'item_embedding', 'prompt_embeddings']
    base_model_list = ['lora_', 'cls', 'classifier', 'LayerNorm']
    for k, v in model.named_parameters():
        if any(key in k for key in key_list) or any(key in k for key in base_model_list):
            v.requires_grad = True
        else:
            v.requires_grad = False


def set_all_parameters_trainable(model):
    for k, v in model.named_parameters():
        v.requires_grad = True


def set_external_trainable_parameters(model):
    key_list = [
        'LayerNorm',
        'classifier',
        'user_embeddings', 'item_embeddings',
        'dense_scale', 'dense_bias',
        'pooler',
        # 'q', 'v'
        # 'q', 'k', 'v',
        'attention',
        # 'out',
        # 'bert',
        "base_model"
    ]
    for name, param in model.named_parameters():
        if any(key in name for key in key_list):
            param.requires_grad = True


def get_predictions(tokenizer, inputs):
    if tokenizer is None:
        return inputs
    else:
        score_list = [str(i) for i in range(10)]
        targets = tokenizer.batch_decode(inputs)
        def get_union(target):
            union_set = set(score_list) & set(target)
            if len(union_set) != 1:
                return -1
            else:
                return int(union_set.pop())
        return [get_union(target) for target in targets]


def build_text_vocab(datasets, tokenizer): # hierarchical documents with multiple datasets.
    from torchtext.vocab import vocab, GloVe
    from collections import Counter
    def load_vectors(itos, vec):
        vectors = []
        for token in itos:
            if token in vec.stoi:
                vectors.append(vec[token])
            else:
                vectors.append(torch.zeros(vec.dim))
        vectors = torch.stack(vectors)
        return vectors
    token_counter = Counter()
    for dataset in datasets:
        for document in dataset:
            for sentence in document.text:
                # tokens = tokenizer(sentence)
                token_counter.update(tokenizer(sentence))

    from collections import OrderedDict
    unk_token = '<unk>'
    pad_token = '<pad>'
    sorted_by_freq_tuples = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    text_vocab = vocab(ordered_dict, specials=[unk_token, pad_token])
    vec = GloVe(name='840B', dim=300)
    vectors = load_vectors(text_vocab.get_itos(), vec)
    return text_vocab, vectors
