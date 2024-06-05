import math
import torch
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from M2A.cfgs.constants import MODEL_MAP

start_map = {
    "bert": "cls_token_id",
    "ma_bert": "cls_token_id",
    "ma_roberta": "cls_token_id",
    "user": "cls_token_id",
    "bert_large": "cls_token_id",
    "roberta": "cls_token_id",
    "roberta_large": "cls_token_id",
    "flan_t5": None,
    "uni_bert": 'cls_token_id',
    "gpt2": None
}

end_map = {
    "bert": "sep_token_id",
    "uni_bert": "sep_token_id",
    "ma_bert": "sep_token_id",
    "ma_roberta": "eos_token_id",
    "user": "sep_token_id",
    "bert_large": "sep_token_id",
    "roberta": "eos_token_id",
    "roberta_large": "eos_token_id",
    "flan_t5": "eos_token_id",
    "gpt2": None,
}
pad_map = {
    "bert": "pad_token_id",
    "uni_bert": "pad_token_id",
    "ma_bert": "pad_token_id",
    "ma_roberta": "pad_token_id",
    "user": "pad_token_id",
    "bert_large": "pad_token_id",
    "roberta": "pad_token_id",
    "roberta_large": "pad_token_id",
    "flan_t5": "pad_token_id",
    "gpt2": "pad_token_id"
}

instruct_text = {
    "bert": None,
    "uni_bert": None,
    "ma_bert": None,
    "user": None,
    "bert_large": None,
    "roberta": None,
    "roberta_large": None,
    "flan_t5": "Review:",
    "ma_roberta": None,
    "gp2": None,
}


instruct_label = {
    "bert": None,
    "uni_bert": None,
    "ma_bert": None,
    "user": None,
    "bert_large": None,
    "roberta": None,
    "roberta_large": None,
    "flan_t5": "sentiment score:",
    "ma_roberta": None,
    "gpt2": None
}

suffix_label = {
    "gpt2": "Please provide the ratings of sentiment is ",
}


def _truncate_and_pad(tokens, start_id=None, end_id=None, pad_id=None, prefix_ids=None, suffix_ids=None, max_length=510, pad_strategy="head"):
    total_length = len(tokens)
    prefix_ids = prefix_ids if prefix_ids is not None else []
    suffix_ids = suffix_ids if suffix_ids is not None else []
    start_id = [start_id] if start_id is not None else []
    end_id = [end_id] if end_id is not None else []
    pad_id = [pad_id] if pad_id is not None else []
    if total_length > max_length:
        if pad_strategy == 'head':
            return start_id + prefix_ids + tokens[:max_length] + suffix_ids + end_id
        if pad_strategy == 'tail':
            return start_id + prefix_ids + tokens[-max_length:] + suffix_ids + end_id
        if pad_strategy == 'both':
            # return [start_id] + tokens[:128] + tokens[-max_length+128:] + [end_id]
            # return prefix_ids + start_id + tokens[:128] + tokens[-max_length+128:] + end_id
            return start_id + prefix_ids + tokens[:max_length//2] + tokens[-max_length+max_length//2:] + suffix_ids + end_id
        return
    else:
        return start_id + prefix_ids + tokens + suffix_ids + end_id + pad_id * (max_length-total_length)


class BucketIteratorForPLMTokenizer(object):
    def __init__(self, data,
                 batch_size,
                 plm_name=None,
                 max_length=512,
                 sort_index=0,
                 shuffle=True,
                 sort=True,
                 device='cpu',
                 trunc=None,
                 description="Train",
                 stoi=None,
                 stoi_p=None,
                 masking_ratio=0.15,
                 ):
        self.shuffle = shuffle
        self.masking_ratio = masking_ratio
        self.stoi = stoi
        self.stoi_p = stoi_p
        self.sort = sort
        self.sort_key = sort_index
        self.max_length = max_length
        self.device = device
        self.description = description
        self.plm_name = plm_name
        self.trunc = trunc
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.plm_name])
        if self.plm_name in ['gpt2']:
            special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.data = data
        self.batch_size = batch_size
        self.batches = self.sort_and_pad(data, batch_size)

        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        dataloader_tqdm = tqdm(range(num_batch))
        dataloader_tqdm.set_description_str("Processing Dataloader: {}".format(self.description))
        for i in dataloader_tqdm:
            if self.plm_name in ['gpt2']:
                batches.append(self.ar_pad_data(sorted_data[i*batch_size: (i+1)*batch_size]))
            else:
                batches.append(self.mlm_pad_data(sorted_data[i*batch_size: (i+1)*batch_size]))

        return batches

    def mlm_pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_prd_indices = []
        batch_labels = []

        max_len_words = max([len(t[0]) for t in batch_data])

        for item in batch_data:
            tokens_index, label, user_index, product_index = item
            if self.trunc:
                tokens_index = _truncate_and_pad(
                    tokens=tokens_index,
                    start_id=getattr(self.tokenizer, start_map[self.plm_name]) if start_map[self.plm_name] is not None else None,
                    end_id=getattr(self.tokenizer, end_map[self.plm_name]) if end_map[self.plm_name] is not None else None,
                    pad_id=getattr(self.tokenizer, pad_map[self.plm_name]),
                    prefix_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instruct_text[self.plm_name]))
                    if instruct_text[self.plm_name] is not None else None,
                    pad_strategy=self.trunc,
                    max_length=max_len_words if max_len_words < self.max_length - 2 else self.max_length - 2
                )
            if instruct_label[self.plm_name] is not None:
                label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instruct_label[self.plm_name] + str(label))) + \
                        [getattr(self.tokenizer, end_map[self.plm_name])]
            batch_text_indices.append(tokens_index)
            batch_labels.append(label)
            batch_usr_indices.append(self.stoi[user_index])
            batch_prd_indices.append(self.stoi_p[product_index])

        if None in batch_labels:
            batch_labels = None
        else:
            batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_usr_indices = torch.tensor(batch_usr_indices, device=self.device)
        batch_prd_indices = torch.tensor(batch_prd_indices, device=self.device)
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)


        return {'input_ids': batch_text_indices,
                'attention_mask': batch_text_indices != getattr(self.tokenizer, pad_map[self.plm_name]),
                'cls_labels': batch_labels,
                'user_ids': batch_usr_indices,
                'item_ids': batch_prd_indices,
                }

    def ar_pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_prd_indices = []
        batch_labels = []
        batch_mlm_labels = []
        batch_valid_len = []

        max_len_words = max([len(t[0]) for t in batch_data])
        for item in batch_data:
            tokens_index, label, user_index, product_index = item
            if len(tokens_index) == 0: # remove blank strings
                continue
            # if len(tokens_index) < self.max_length:
            #     validate_len_token = len(tokens_index)
            # else:
            #     validate_len_token = self.max_length
            validate_len_token = len(tokens_index) if len(tokens_index) < self.max_length else self.max_length
            # label_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(suffix_label[self.plm_name]) + str(label))
            # validate_lab_token = len(tokens_index) + len(label_ids)
            suffix_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(suffix_label[self.plm_name]))
            if self.trunc:
                tokens_index = _truncate_and_pad(
                    tokens=tokens_index,
                    suffix_ids=suffix_ids,
                    pad_id=getattr(self.tokenizer, pad_map[self.plm_name]) if pad_map[self.plm_name] is not None else None,
                    start_id=getattr(self.tokenizer, start_map[self.plm_name]) if start_map[self.plm_name] is not None else None,
                    end_id=getattr(self.tokenizer, end_map[self.plm_name]) if end_map[self.plm_name] is not None else None,
                    pad_strategy=self.trunc,
                    max_length=max_len_words if max_len_words < self.max_length else self.max_length
                )

            # shift labels
            shift_labels = tokens_index[1:validate_len_token] + [-100] * (len(tokens_index) - (validate_len_token-1))
            batch_text_indices.append(tokens_index)
            batch_labels.append(label)
            batch_usr_indices.append(self.stoi[user_index])
            batch_prd_indices.append(self.stoi_p[product_index])
            batch_mlm_labels.append(shift_labels)
            batch_valid_len.append(validate_len_token+len(suffix_ids))

        if None in batch_labels:
            batch_labels = None
        else:
            batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_usr_indices = torch.tensor(batch_usr_indices, device=self.device)
        batch_mlm_labels = torch.tensor(batch_mlm_labels, device=self.device)
        batch_prd_indices = torch.tensor(batch_prd_indices, device=self.device)
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_valid_len = torch.tensor(batch_valid_len, device=self.device)

        return {'input_ids': batch_text_indices,
                'attention_mask': batch_text_indices != getattr(self.tokenizer, pad_map[self.plm_name]),
                'cls_labels': batch_labels,
                'mlm_labels': batch_mlm_labels,
                'user_ids': batch_usr_indices,
                'item_ids': batch_prd_indices,
                'valid_len': batch_valid_len,
                }

    def _generate_masked_inputs(self):
        for batch in self.batches:
            batch_input_masked = []
            batch_input_masked_labels = []
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            mask_token_idx = \
            torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.tokenizer.mask_token)),
                         device=self.device)[0]
            for input_id, att in zip(input_ids, attention_mask):
                input_length = sum(att)
                mask_input = input_id[:input_length]
                tokens_padding = input_id[input_length:]

                mask = torch.zeros(input_length, device=self.device)
                mask_num = max(int(input_length * self.masking_ratio), 1)
                random_list = [random.randint(1, input_length - 2) for _ in range(mask_num)]
                for idx in random_list:
                    mask[idx] = 1.0
                masked_inputs = mask_input.masked_fill(mask == 1, mask_token_idx)
                masked_inputs = torch.cat((masked_inputs, tokens_padding), dim=-1)

                mask_labels = input_id.masked_fill(masked_inputs != mask_token_idx, -100)
                batch_input_masked.append(masked_inputs)
                batch_input_masked_labels.append(mask_labels)
            batch_input_masked = torch.stack(batch_input_masked, dim=0)
            batch_input_masked_labels = torch.stack(batch_input_masked_labels, dim=0)
            batch["mask_input_ids"] = batch_input_masked
            batch["mlm_labels"] = batch_input_masked_labels


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
            # self._generate_masked_inputs()
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len


class BucketIteratorForDecoderPLMs(object):
    def __init__(self, data,
                 batch_size,
                 plm_name=None,
                 max_length=512,
                 sort_index=0,
                 shuffle=True,
                 sort=True,
                 device='cpu',
                 trunc=None,
                 description="Train",
                 stoi=None,
                 stoi_p=None,
                 ):
        self.shuffle = shuffle
        self.stoi = stoi
        self.stoi_p = stoi_p
        self.sort = sort
        self.sort_key = sort_index
        self.max_length = max_length
        self.device = device
        self.description = description
        self.plm_name = plm_name
        self.trunc = trunc
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.plm_name])
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.data = data
        self.batch_size = batch_size
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        dataloader_tqdm = tqdm(range(num_batch))
        dataloader_tqdm.set_description_str("Processing Dataloader: {}".format(self.description))
        for i in dataloader_tqdm:
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_prd_indices = []
        batch_labels = []
        batch_mlm_labels = []

        max_len_words = max([len(t[0]) for t in batch_data])

        for item in batch_data:
            tokens_index, label, user_index, product_index = item
            validate_len_token = len(tokens_index)
            # label_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(suffix_label[self.plm_name]) + str(label))
            # validate_lab_token = len(tokens_index) + len(label_ids)
            suffix_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(suffix_label[self.plm_name]))
            if self.trunc:
                tokens_index = _truncate_and_pad(
                    tokens=tokens_index,
                    suffix_ids=suffix_ids,
                    pad_id=getattr(self.tokenizer, pad_map[self.plm_name]) if pad_map[self.plm_name] is not None else None,
                    start_id=getattr(self.tokenizer, start_map[self.plm_name]) if start_map[self.plm_name] is not None else None,
                    end_id=getattr(self.tokenizer, end_map[self.plm_name]) if end_map[self.plm_name] is not None else None,
                    pad_strategy=self.trunc,
                    max_length=max_len_words if max_len_words < self.max_length - 2 else self.max_length - 2
                )

            # shift labels
            shift_labels = tokens_index[1:validate_len_token] + [-100] * (len(tokens_index) - (validate_len_token-1))
            batch_text_indices.append(tokens_index)
            batch_labels.append(label)
            batch_usr_indices.append(self.stoi[user_index])
            batch_prd_indices.append(self.stoi_p[product_index])
            batch_mlm_labels.append(shift_labels)

        batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_usr_indices = torch.tensor(batch_usr_indices, device=self.device)
        batch_mlm_labels = torch.tensor(batch_mlm_labels, device=self.device)
        batch_prd_indices = torch.tensor(batch_prd_indices, device=self.device)
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)

        return {'input_ids': batch_text_indices,
                'attention_mask': batch_text_indices != getattr(self.tokenizer, pad_map[self.plm_name]),
                'cls_labels': batch_labels,
                'mlm_labels': batch_mlm_labels,
                'user_ids': batch_usr_indices,
                # 'item_ids': batch_prd_indices,
                'item_ids': None,
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len
