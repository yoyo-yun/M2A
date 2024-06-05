import torch
from peft.utils import _get_submodules
from M2A.models.plms.LoRA import OurLinear, LoraConfig
# from models.plms.gpt2 import GPT2ForUnifiedLM
from M2A.models.plms.gpt2_new import GPT2ForUnifiedLM
from peft.tuners.tuners_utils import check_target_module_exists
from M2A.cfgs.constants import MODEL_MAP
from transformers import AutoTokenizer

Checkpoint_map = {
    "gpt2": GPT2ForUnifiedLM,
}


class BayesianUnifiedDecoderModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.usr_dim < 0:
            self.register_module("user_embedding", None)
        else:
            self.user_embedding = torch.nn.Embedding(self.config.usr_size, self.config.usr_dim,
                                                     _weight=torch.zeros(self.config.usr_size, self.config.usr_dim),
                                                     _freeze=False)
        if self.config.usr_dim < 0:
            self.register_module("item_embedding", None)
        else:
            self.item_embedding = torch.nn.Embedding(self.config.prd_size, self.config.prd_dim,
                                                     _weight=torch.zeros(self.config.prd_size, self.config.prd_dim),
                                                     _freeze=False)
        self.base_model = Checkpoint_map[config.model].from_pretrained(
            MODEL_MAP[config.model],
            return_dict=True,
            num_labels=self.config.num_labels, )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.config.model])
        if self.config.model in ['gpt2']:
            special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
            tokenizer.add_special_tokens(special_tokens_dict)
            self.base_model.resize_token_embeddings(len(tokenizer))
            self.base_model.config.pad_token_id = tokenizer.pad_token_id
        self.get_word_embedding()  # self.word_embeddings
        self.peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.1,
                                      user=True, item=True,
                                      target_modules=["c_attn"],
                                      )
        self.peft(self.peft_config)  # for bert

    def forward(self,
                input_ids=None,
                mask_input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                cls_labels=None,
                mlm_labels=None,
                user_ids=None,
                item_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm=False,
                cls=True,
                valid_len=None,
                global_version=False,
                **kwargs, ):
        user_ids = user_ids if self.user_embedding is not None else None
        item_ids = item_ids if self.item_embedding is not None else None

        input_ids, mlm_labels, cls_labels = self.initial_states(mlm, cls, mask_input_ids, mlm_labels, input_ids, cls_labels)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if kwargs.get("position_ids", None) is not None:
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "cls_labels": cls_labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "valid_len": valid_len,
            }
        )
        self.update_input(user_ids, item_ids, global_version, kwargs)
        return self.base_model(inputs_embeds=inputs_embeds, mlm_labels=mlm_labels, **kwargs)

    def initial_states(self, mlm, cls, mask_input_ids, mlm_labels, input_ids, cls_labels):
        if mlm and cls:  # set training both targets
            mlm_labels = mlm_labels
            cls_labels = cls_labels
        elif mlm:  # set training mlm
            mlm_labels = mlm_labels
            cls_labels = None
        elif cls:  # set training cls
            mlm_labels = None
            cls_labels = cls_labels

        if not self.training:
            # input_ids = input_ids
            mlm_labels = None
            cls_labels = None
        return input_ids, mlm_labels, cls_labels

    # without prompt, putting user and item embedding into nodes_hidden_states
    def update_input(self, user_ids, item_ids, global_version, kwargs):
        if not global_version:
            if user_ids is not None or item_ids is not None:
                if user_ids is not None and item_ids is not None:
                    user_prompts = self.user_embedding(user_ids).unsqueeze(1)
                    item_prompts = self.item_embedding(item_ids).unsqueeze(1)
                    kwargs.update({"nodes_hidden_states": (user_prompts, item_prompts)})
                else:
                    if user_ids is not None:
                        user_prompts = self.user_embedding(user_ids).unsqueeze(1)
                        kwargs.update({"nodes_hidden_states": (user_prompts, None)})
                    if item_ids is not None:
                        item_prompts = self.item_embedding(item_ids).unsqueeze(1)
                        kwargs.update({"nodes_hidden_states": (None, item_prompts)})
        else:
            if user_ids is not None or item_ids is not None:
                user_ids, item_ids = None, None
                if user_ids is not None and item_ids is not None:
                    user_prompts = self.general_user_embedding(user_ids).unsqueeze(1)
                    item_prompts = self.general_item_embedding(item_ids).unsqueeze(1)
                    kwargs.update({"nodes_hidden_states": (user_prompts, item_prompts)})
                else:
                    if user_ids is not None:
                        user_prompts = self.general_user_embedding(user_ids).unsqueeze(1)
                        kwargs.update({"nodes_hidden_states": (user_prompts, None)})
                    if item_ids is not None:
                        item_prompts = self.general_item_embedding(item_ids).unsqueeze(1)
                        kwargs.update({"nodes_hidden_states": (None, item_prompts)})


    def peft(self, lora_config):
        key_list = [key for key, _ in self.base_model.named_modules()]
        for key in key_list:
            if not check_target_module_exists(lora_config, key):
                continue
            parent, target, target_name = _get_submodules(self.base_model, key)
            target.injecting_parameters({
                "r": lora_config.r,
                "user": lora_config.user,
                "item": lora_config.item,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "bias": lora_config.bias
            })
        self._mark_only_adapters_as_trainable()

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.base_model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        bias = self.peft_config.bias
        if bias == "all":
            for n, p in self.base_model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in self.base_model.modules():
                if isinstance(m, OurLinear) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            pass
            # raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def get_word_embedding(self):
        for named_param, value in list(self.base_model.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = self.base_model.get_submodule(named_param.replace(".weight", ""))
                break


    def get_cal_regularization(self):
        weights = []  # [(bs, in_dim, out_dim)]
        key_list = [key for key, _ in self.base_model.named_modules()]
        for key in key_list:
            if not check_target_module_exists(self.peft_config, key):
                continue
            parent, target, target_name = _get_submodules(self.base_model, key)
            if target.regularization_weights is not None:
                weights.append(target.regularization_weights)
                target.regularization_weights = None
        if len(weights) == 0:
            return None
        weights = torch.stack(weights, dim=1) # (bs, ~, in_dim, out_dim)
        loss = torch.nn.MSELoss(reduction='mean')(weights, torch.zeros_like(weights, device=weights.device))
        return loss