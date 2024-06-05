import os
import torch
import datetime
import evaluate
from M2A.models import *
from tqdm import tqdm
from M2A.trainer.optimizer import get_AdamW_optim, get_Adam_optim_linear
from M2A.trainer.utils import load_dataset_whole_document_plms,\
    Metrics, print_trainable_parameters, get_predictions, set_trainable
from transformers import AutoTokenizer

ALL_MODLES = {
    "bert": BertForSequenceClassification,
    "bert_large": BertForSequenceClassification,
    "roberta": BayesianUnifiedModel,
    'gpt2': BayesianUnifiedDecoderModel,
}


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.best_dev_acc_general = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.log_path = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt'
        self.train_metrics = Metrics([evaluate.load(path=f"metrics/{name}.py") for name in config.metrics_list])
        self.dev_metrics = Metrics([evaluate.load(path=f"metrics/{name}.py") for name in config.metrics_list])

    def train(self):
        pass

    def train_epoch(self):
        pass

    def eval(self, eval_itr, state=None):
        pass

    def empty_log(self):
        if (os.path.exists(self.log_path)): os.remove(self.log_path)
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def resume_log(self):
        # Save log information
        logfile = open(self.log_path, 'a+')
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.write(str(self.config))
        logfile.write('\n')
        logfile.close()

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, results: dict, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n"
        for k, v in results.items():
            logs = logs + k + ": " + "{:.4f} ".format(v)
        logs = logs + "\n"
        return logs

    def ensureDirs(self, *dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


from M2A.trainer.bucket_iteractor import MODEL_MAP, pad_map, _truncate_and_pad, start_map, end_map
class predictor:
    def __init__(self, data, stoi, plm_name, device='cuda'):
        self.data = data
        self.device = device
        self.stoi = stoi
        self.plm_name = plm_name
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.plm_name])
        self.batches = self.create_data(self.data)

    def create_data(self, data):
        batches = []
        for item in data:
            tokens_index, label, user_index, product_index = item
            if len(tokens_index) > 510:
                tokens_index = _truncate_and_pad(
                    tokens=tokens_index,
                    start_id=getattr(self.tokenizer, start_map[self.plm_name]) if start_map[
                                                                                      self.plm_name] is not None else None,
                    end_id=getattr(self.tokenizer, end_map[self.plm_name]) if end_map[self.plm_name] is not None else None,
                    pad_id=getattr(self.tokenizer, pad_map[self.plm_name]),
                    prefix_ids=None,
                    pad_strategy="both",
                    max_length=510
                )
            data_sample = {
                'input_ids': torch.tensor([tokens_index], device=self.device),
                'attention_mask': torch.tensor([tokens_index], device=self.device) != getattr(self.tokenizer, pad_map[self.plm_name]),
                'cls_labels': torch.tensor([label], device=self.device),
                'p': torch.tensor([self.stoi[user_index]], device=self.device)
            }
            batches.append(data_sample)
        return batches

    def __iter__(self):
        for idx in range(len(self.batches)):
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


class BayesianTrainer(Trainer):
    def __init__(self, config):
        super(BayesianTrainer, self).__init__(config)
        # loading dataloader
        from_scratch = False
        if self.config.dataset in ["mtl"]:
            from_scratch = True
        self.train_itr, self.dev_itr, self.test_itr, self.un_itr, self.usr_stoi, _ = \
            load_dataset_whole_document_plms(self.config, from_scratch=from_scratch)
        self.moniter_per_step = len(self.train_itr) // self.config.num_monitor_times_per_epoch
        training_steps_per_epoch = len(self.train_itr) // (self.config.gradient_accumulation_steps)
        self.config.num_train_optimization_steps = self.config.TRAIN.max_epoch * training_steps_per_epoch
        self.best_checkpoint = None

    def set_FFT(self):
        model_name_or_path = MODEL_MAP[self.config.model]
        print(ALL_MODLES[self.config.model])
        # self.net = ALL_MODLES[self.config.model](self.config).to(self.config.device)
        self.net = BayesianUnifiedModel(self.config).to(self.config.device)

        for k, v in self.net.named_parameters():
            v.requires_grad = True
        print_trainable_parameters(self.net, verbose=False)

        self.config.TRAIN.lr_base = 2e-5
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        # self.optim, self.scheduler = get_Adam_optim_linear(self.config, self.net)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if self.config.model in ['flan_t5'] else None

    def set_global_version(self):
        self.load_state(self.config.dataset)
        self.net.general_user_embedding = torch.nn.Embedding(1, self.config.usr_dim, _weight=torch.zeros(1, self.config.usr_dim),
                                                         _freeze=False
                                                         ).to(self.config.device)
        self.net.general_item_embedding = torch.nn.Embedding(1, self.config.usr_dim, _weight=torch.zeros(1, self.config.usr_dim),
                                                         _freeze=False
                                                         ).to(self.config.device)

        for k, v in self.net.named_parameters():
            if "general_" in k:
                v.requires_grad = True
            else:
                v.requires_grad = False

        print_trainable_parameters(self.net, verbose=True)
        self.config.TRAIN.lr_base = 2e-4
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        self.tokenizer = None

    def set_AR_FFT(self):
        model_name_or_path = MODEL_MAP[self.config.model]
        print(ALL_MODLES[self.config.model])
        # self.net = ALL_MODLES[self.config.model](self.config).to(self.config.device)
        self.net = BayesianUnifiedDecoderModel(self.config).to(self.config.device)

        for k, v in self.net.named_parameters():
            v.requires_grad = True
        print_trainable_parameters(self.net, verbose=False)

        self.config.TRAIN.lr_base = 2e-5
        # self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        self.optim, self.scheduler = get_Adam_optim_linear(self.config, self.net)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if self.config.model in [
            'flan_t5'] else None

    def set_PEFT(self):
        model_name_or_path = MODEL_MAP[self.config.model]
        print(ALL_MODLES[self.config.model])
        self.net = BayesianUnifiedModel(self.config).to(self.config.device)

        set_trainable(self.net)
        print_trainable_parameters(self.net, verbose=False)

        # self.config.TRAIN.lr_base = 1e-4 # for yelp_13
        # self.config.TRAIN.lr_base = 9e-5 # for yelp_14
        self.config.TRAIN.lr_base = 2e-5
        # self.optim, self.scheduler = get_Adam_optim_linear(self.config, self.net)
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if self.config.model in ['flan_t5'] else None

    def joint_training(self, start_monitor_epoch=1):
        for epoch in range(1, self.config.TRAIN.max_epoch + 1):
            self.net.train()
            train_results, dev_results_single, dev_results_general = self.joint_train_epoch(epoch,
                                                          mlm=True,
                                                          cls=True,
                                                          start_monitor_epoch=start_monitor_epoch)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_results, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.log_path, logs)

            # logging evaluating logs
            if dev_results_single is not None:
                eval_logs = self.get_logging(dev_results_single, eval="evaluating")
                print("\r" + eval_logs)
                self.logging(self.log_path, eval_logs)
                if dev_results_general is not None:
                    eval_logs = self.get_logging(dev_results_general, eval="general evaluating")
                    print("\r" + eval_logs)
                    self.logging(self.log_path, eval_logs)

                # early stopping
                if dev_results_single['accuracy'] < self.best_dev_acc:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.log_path + "\n" + \
                                          "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                        print(early_stop_logs)
                        self.logging(self.log_path, early_stop_logs)
                        break
                else:
                    self.unimproved_iters = 0
                    self.best_dev_acc = dev_results_single['accuracy']

    def joint_train_epoch(self, epoch=1, mlm=False, cls=False, start_monitor_epoch=2):
        eval_best_acc = 0.
        eval_best_metrics = None
        dev_metrics_general = None
        if mlm and self.config.model not in ['gpt2']:
            print("Generating randomly Mask tokens..."); self.train_itr._generate_masked_inputs(); print("Done!")
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {} with MLM->{} & CLS->{}".format(epoch, mlm, cls))
        self.optim.zero_grad()
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            outputs_single = self.net(**batch, mlm=mlm, cls=cls, global_version=False)
            mlm_loss = outputs_single.mlm_loss
            cls_loss = outputs_single.cls_loss
            epoch_tqdm.set_postfix(mlm_loss=mlm_loss.item() if mlm_loss is not None else mlm_loss,
                                   cls_loss=cls_loss.item() if cls_loss is not None else cls_loss)
            outputs_general = self.net(**batch, mlm=mlm, cls=cls, global_version=True)
            logits_single = outputs_single.logits
            logits_general = outputs_general.logits
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False) \
                (torch.log_softmax(logits_single / 4, dim=-1), torch.softmax(logits_general / 4, dim=-1))
            loss = outputs_single.loss + outputs_general.loss * 0.5
            # loss = outputs_single.loss + kl_loss
            # loss = outputs_single.loss
            if self.config.gradient_accumulation_steps >= 2:
                loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                # torch.nn.utils.clip_grad_norm_(self.net.PARAMETERS(), 2) # for yelp14
                self.optim.step()
                if self.scheduler is not None: self.scheduler.step()
                self.optim.zero_grad()
            predictions = outputs_single.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["cls_labels"])
            self.train_metrics.add_batch(predictions=predictions, references=references)

            # if step % self.moniter_per_step == 0 and step != 0:
            if step % self.moniter_per_step == 0 and step != 0 and epoch >= start_monitor_epoch and cls:
                self.net.eval()
                with torch.no_grad():
                    dev_metrics = self.eval(self.dev_itr, global_version=False)

                # monitoring eval metrics
                if dev_metrics["accuracy"] > eval_best_acc:
                    eval_best_acc = dev_metrics["accuracy"]
                    eval_best_metrics = dev_metrics
                    if dev_metrics["accuracy"] > self.best_dev_acc:
                        # saving models
                        self.best_dev_acc = dev_metrics["accuracy"]
                        with torch.no_grad():
                            dev_metrics_general = self.eval(self.dev_itr, global_version=True)
                        self.best_dev_acc_general = dev_metrics_general["accuracy"] # for continue learning
                        self.saving_model()

        return self.train_metrics.compute(average="macro"), eval_best_metrics, dev_metrics_general

    def continue_learning(self, start_monitor_epoch=1):
        # set trainable for only coarse views
        key_list = ['lora_UA', 'lora_UB', 'lora_IA', 'lora_IB']
        for k, v in self.net.named_parameters():
            if any(key in k for key in key_list):
                v.requires_grad = True
            else:
                v.requires_grad = False

        print_trainable_parameters(self.net, False)
        # optim_states = self.optim.state_dict()
        self.config.TRAIN.lr_base = 2e-5
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)

        # initialize continue learning settings
        # self.optim.load_state_dict(optim_states)
        self.unimproved_iters = 0

        for epoch in range(1, self.config.TRAIN.max_epoch + 1):
            self.net.train()
            train_results, dev_results_general = self.continue_learning_epoch(epoch,
                                                                              mlm=True,
                                                                              cls=True,
                                                                              start_monitor_epoch=start_monitor_epoch)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_results, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.log_path, logs)

            # logging evaluating logs
            if dev_results_general is not None:
                eval_logs = self.get_logging(dev_results_general, eval="general evaluating")
                print("\r" + eval_logs)
                self.logging(self.log_path, eval_logs)

                # early stopping
                if dev_results_general['accuracy'] < self.best_dev_acc_general:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.log_path + "\n" + \
                                          "Early Stopping. Epoch: {}, Best General Dev Acc: {}".format(epoch, self.best_dev_acc_general)
                        print(early_stop_logs)
                        self.logging(self.log_path, early_stop_logs)
                        break
                else:
                    self.unimproved_iters = 0
                    self.best_dev_acc_general = dev_results_general['accuracy']

    def continue_learning_epoch(self, epoch=1, mlm=False, cls=False, start_monitor_epoch=1):
        eval_best_acc_general = 0.
        eval_best_metrics_general = None
        dev_metrics_general = None
        if mlm and self.config.model not in ['gpt2']:
            print("Generating randomly Mask tokens...")
            self.train_itr._generate_masked_inputs()
            print("Done!")
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {} with MLM->{} & CLS->{}".format(epoch, mlm, cls))
        self.optim.zero_grad()
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            outputs_general = self.net(**batch, mlm=mlm, cls=cls, global_version=True)
            mlm_loss = outputs_general.mlm_loss
            cls_loss = outputs_general.cls_loss
            epoch_tqdm.set_postfix(mlm_loss=mlm_loss.item() if mlm_loss is not None else mlm_loss,
                                   cls_loss=cls_loss.item() if cls_loss is not None else cls_loss)
            loss = outputs_general.loss * 0.5
            if self.config.gradient_accumulation_steps >= 2:
                loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                self.optim.step()
                if self.scheduler is not None: self.scheduler.step()
                self.optim.zero_grad()
            predictions = outputs_general.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions),\
                                      get_predictions(self.tokenizer, batch["cls_labels"])
            self.train_metrics.add_batch(predictions=predictions, references=references)

            # if step % self.moniter_per_step == 0 and step != 0:
            if step % self.moniter_per_step == 0 and step != 0 and epoch >= start_monitor_epoch and cls:
                self.net.eval()
                with torch.no_grad():
                    dev_metrics_general = self.eval(self.dev_itr, global_version=True)

                # monitoring eval metrics
                if dev_metrics_general["accuracy"] > eval_best_acc_general:
                    eval_best_acc_general = dev_metrics_general["accuracy"]
                    eval_best_metrics_general = dev_metrics_general
                    if eval_best_metrics_general["accuracy"] > self.best_dev_acc_general:
                        # saving models
                        self.best_dev_acc_general = eval_best_metrics_general["accuracy"]
                        self.saving_model()

        return self.train_metrics.compute(average="macro"), eval_best_metrics_general

    def train_epoch(self, epoch=1, mlm=False, cls=False, start_monitor_epoch=2):
        eval_best_acc = 0.
        eval_best_metrics = None
        if mlm and self.config.model not in ['gpt2']:
            print("Generating randomly Mask tokens..."); self.train_itr._generate_masked_inputs(); print("Done!")
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {} with MLM->{} & CLS->{}".format(epoch, mlm, cls))
        self.optim.zero_grad()
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            # batch['user_ids'] = None; batch['item_ids'] = None
            outputs = self.net(**batch, mlm=mlm, cls=cls)
            mlm_loss = outputs.mlm_loss
            cls_loss = outputs.cls_loss
            reg_loss = self.net.get_cal_regularization()
            epoch_tqdm.set_postfix(mlm_loss=mlm_loss.item() if mlm_loss is not None else mlm_loss,
                                   cls_loss=cls_loss.item() if cls_loss is not None else cls_loss,
                                   reg_loss=reg_loss.item() if reg_loss is not None else reg_loss,)
            loss = outputs.loss
            # loss = outputs.loss + reg_loss*0.5
            if self.config.gradient_accumulation_steps >= 2:
                loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0) # for yelp_13
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2) # for yelp14
                self.optim.step()
                if self.scheduler is not None: self.scheduler.step()
                self.optim.zero_grad()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["cls_labels"])
            self.train_metrics.add_batch(predictions=predictions, references=references, )

            # if step % self.moniter_per_step == 0 and step != 0:
            if step % self.moniter_per_step == 0 and step != 0 and epoch >= start_monitor_epoch and cls:
                self.net.eval()
                with torch.no_grad():
                    dev_metrics = self.eval(self.dev_itr)

                # monitoring eval metrics
                if dev_metrics["accuracy"] > eval_best_acc:
                    eval_best_acc = dev_metrics["accuracy"]
                    eval_best_metrics = dev_metrics
                    if dev_metrics["accuracy"] > self.best_dev_acc:
                        # saving models
                        self.best_dev_acc = dev_metrics["accuracy"]
                        # self.best_checkpoint = self.net
                        # with torch.no_grad():
                        #     print(self.eval(self.test_itr, zero_shot))
                        self.saving_model()

        return self.train_metrics.compute(average="macro"), eval_best_metrics

    def train(self, start_monitor_epoch=1):
        num_mlm_iterator = 3
        # cls_at_every_iterator = True
        cls_at_every_iterator = False
        cls_list = []
        mlm_list = []
        for i in range(1, self.config.TRAIN.max_epoch+1):
            if i % num_mlm_iterator == 0:
                cls_list.append(i)
            else:
                if cls_at_every_iterator: cls_list.append(i)
                mlm_list.append(i)

        print("cls training epochs are {}".format(cls_list))
        print("mlm training epochs are {}".format(mlm_list))
        for epoch in range(1, self.config.TRAIN.max_epoch + 1):
            self.net.train()
            train_results, dev_results = self.train_epoch(epoch,
                                                          # mlm=epoch in mlm_list,
                                                          # cls=epoch in cls_list,
                                                          mlm=False,
                                                          cls=True,
                                                          start_monitor_epoch=start_monitor_epoch)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_results, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.log_path, logs)

            # logging evaluating logs
            if dev_results is not None:
                eval_logs = self.get_logging(dev_results, eval="evaluating")
                print("\r" + eval_logs)
                self.logging(self.log_path, eval_logs)

                # early stopping
                if dev_results['accuracy'] < self.best_dev_acc:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.log_path + "\n" + \
                                          "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                        print(early_stop_logs)
                        self.logging(self.log_path, early_stop_logs)
                        break
                else:
                    self.unimproved_iters = 0
                    self.best_dev_acc = dev_results['accuracy']

    def saving_model(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.ensureDirs(os.path.join(SAVED_MODEL_PATH, self.config.dataset, self.config.model))
        path = os.path.join(SAVED_MODEL_PATH, self.config.dataset, self.config.model, "ckpt.pkl")
        torch.save(self.net.state_dict(), path)

    def load_state(self, dataset, only_base_model=False):
        if self.config.model in ['gpt2']:
            self.net = BayesianUnifiedDecoderModel(self.config).to(self.config.device)
        else:
            self.net = BayesianUnifiedModel(self.config).to(self.config.device)
        SAVED_MODEL_PATH = self.config.ckpts_path
        path = os.path.join(SAVED_MODEL_PATH, dataset, self.config.model, 'ckpt.pkl')
        self.net.load_state_dict(torch.load(path), False)
        model_name_or_path = MODEL_MAP[self.config.model]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if self.config.model in [
            'flan_t5'] else None

    def eval(self, eval_itr, zero_shot=False, global_version=False):
        self.net.eval()
        for step, batch in enumerate(eval_itr):
            if zero_shot: batch['user_ids'] = None; batch['item_ids'] = None
            outputs = self.net(**batch, global_version=global_version)
            reg_loss = self.net.get_cal_regularization()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["cls_labels"])
            self.dev_metrics.add_batch(predictions=predictions, references=references,)
        return self.dev_metrics.compute(average="macro")

    def random_eval(self, eval_itr):
        self.net.eval()
        for step, batch in enumerate(eval_itr):
            batch['user_ids'] = torch.randint(0, self.config.usr_size, batch['user_ids'].size(), device=batch['user_ids'].device)
            batch['item_ids'] = torch.randint(0, self.config.prd_size, batch['item_ids'].size(), device=batch['item_ids'].device)
            outputs = self.net(**batch)
            reg_loss = self.net.get_cal_regularization()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["cls_labels"])
            self.dev_metrics.add_batch(predictions=predictions, references=references, )
        return self.dev_metrics.compute(average="macro")

    def eval_domain(self, eval_itr, zero_shot=False):
        self.net.eval()
        domain_list = ["books","electronics","dvd","kitchen_housewares",
                       "apparel","camera_photo","health_personal_care","music",
                       "toys_games","video","baby","magazines",
                       "software", "sports_outdoors","imdb", "MR",]
        dev_metrics_list = [
            Metrics([evaluate.load(path=f"metrics/{name}.py") for name in self.config.metrics_list])
            for _ in range(len(domain_list))]
        for step, batch in enumerate(eval_itr):
            if zero_shot: user_ids = batch.pop('user_ids')
            else: user_ids = batch['user_ids']
            outputs = self.net(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["cls_labels"])
            for u, p, l in zip(user_ids, predictions, batch["cls_labels"]):
                dev_metrics_list[u.item()].add_batch(predictions=[p], references=[l])

        # test_domain_list = ["apparel"]
        test_domain_list = domain_list
        for domain in test_domain_list:
            print(domain + ":\t", end="")
            print(dev_metrics_list[self.usr_stoi[domain]].compute(average="macro")["accuracy"])
        # for metric in dev_metrics_list:
        #     print(metric.compute(average="macro"))
        # print(self.usr_stoi)
        # return self.dev_metrics.compute(average="macro")

    def finding_case(self, eval_itr):
        import csv

        self.net.eval()
        golden_label = []
        predicted_label = []
        predicted_attentions_list = []
        general_attentions_list = []
        general_label = []
        input_text = []
        cnt = 0
        # 文件写入
        with open('case_study.csv', 'w') as csvfile:
            # 初始化写入对象
            f_csv_wt = csv.writer(csvfile)
            # 写入字典格式数据
            filetags = ['id', 'golden_labels', 'pre_label', 'general_label', 'text', 'user_id', 'item_id']
            writer = csv.DictWriter(csvfile, fieldnames=filetags)
            writer.writeheader()  # 定义文件title头信息
            for step, batch in enumerate(eval_itr):
                batch.update({"output_attentions": True})
                outputs = self.net(**batch, global_version=False)
                logits = outputs.logits
                pred_label = torch.argmax(logits, dim=1)
                attentions = outputs.attentions

                general_outputs = self.net(**batch, global_version=True)
                general_logits = general_outputs.logits
                general_pred_label = torch.argmax(general_logits, dim=1)
                general_attentions = general_outputs.attentions

                if pred_label != general_pred_label:
                    # golden_label.append(batch["cls_labels"].item())
                    # predicted_label.append(pred_label.item())
                    # general_label.append(general_pred_label.item())
                    text = self.tokenizer.decode(batch["input_ids"][0])
                    # input_text.append(text)
                    # predicted_attentions_list.append(attentions)
                    # general_attentions_list.append(general_attentions)
                    # print("=" * 10)
                    # print(batch["cls_labels"].item())
                    # print(pred_label)
                    # print(general_pred_label)
                    # print(text)
                    # writerow  写入
                    # f_csv_wt.writerow({'id': cnt,
                    #                    'golden_labels': batch["cls_labels"].item(),
                    #                    'pre_label': pred_label.item(),
                    #                    'general_label': general_pred_label.item(),
                    #                    # 'text': text,
                    #                    # 'attention': attentions,
                    #                    # 'general_attention': general_attentions
                    #                    })
                    f_csv_wt.writerow([cnt,
                                       batch["cls_labels"].cpu().item(),
                                       pred_label.cpu().item(),
                                       general_pred_label.cpu().item(),
                                       text,
                                       batch["user_ids"].cpu().item(),
                                       batch["item_ids"].cpu().item(),
                                       ])
                    cnt += 1

    def case_2(self):
        sample = "salad and bread delicious. entrees just ok."
        user_id = 887,
        item_id = 284,
        # rand_user_id = random.randint(0, len(self.usr_stoi)),
        # print(rand_user_id)
        rand_user_id = 111

        input_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sample))
        print(self.tokenizer.tokenize(sample))
        input_ids = [getattr(self.tokenizer, "cls_token_id")] + input_id + [getattr(self.tokenizer, "eos_token_id")]
        # attention_mask = input_ids != getattr(self.tokenizer, "pad_token_id")
        input_ids = torch.tensor([input_ids], device=self.config.device)
        user_ids = torch.tensor([user_id], device=self.config.device)
        item_ids = torch.tensor([item_id], device=self.config.device)
        random_user_ids = torch.tensor([rand_user_id], device=self.config.device)
        # print(input_ids)
        # print((input_ids != getattr(self.tokenizer, "pad_token_id")))
        batch = {"input_ids": input_ids,
                 "attention_mask": input_ids != getattr(self.tokenizer, "pad_token_id"),
                 "user_ids": user_ids,
                 "item_ids": item_ids,
                 "output_attentions": True,
                 }
        outputs = self.net(**batch, global_version=False)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=1)
        attentions = outputs.attentions

        general_outputs = self.net(**batch, global_version=True)
        general_logits = general_outputs.logits
        general_pred_label = torch.argmax(general_logits, dim=1)
        general_attentions = general_outputs.attentions

        batch.update({"user_ids": random_user_ids,})
        f_outputs = self.net(**batch, global_version=False)
        f_logits = f_outputs.logits
        f_pred_label = torch.argmax(f_logits, dim=1)
        f_attentions = f_outputs.attentions

        # print(attentions)
        # print(general_attentions)
        # print(f_attentions)
        print(pred_label)
        print(general_pred_label)
        print(f_pred_label)

        # original attentions
        if self.config.model in ['gpt2']:
            self.net = BayesianUnifiedDecoderModel(self.config).to(self.config.device)
        else:
            self.net = BayesianUnifiedModel(self.config).to(self.config.device)

        batch = {"input_ids": input_ids,
                 "attention_mask": input_ids != getattr(self.tokenizer, "pad_token_id"),
                 "output_attentions": True,
                 }
        outputs = self.net(**batch, global_version=False)
        o_attentions = outputs.attentions

        # saving attentions
        all_attentions = {
            "attentions": attentions,
            "g_attentions": general_attentions,
            "f_attentions": f_attentions,
            "o_attentions": o_attentions
        }
        torch.save(all_attentions, "case_attentions.pkl")

        def convert_cuda_into_cpus(all_attentions):
            import torch
            attentions, g_attentions, f_attentions, o_attentions = (), (), (), ()

            for i in all_attentions["attentions"]:
                attentions += (i.cpu(),)
            for i in all_attentions["g_attentions"]:
                g_attentions += (i.cpu(),)
            for i in all_attentions["f_attentions"]:
                f_attentions += (i.cpu(),)
            for i in all_attentions["o_attentions"]:
                o_attentions += (i.cpu(),)

            # saving attentions
            all_attentions = {
                "attentions": attentions,
                "g_attentions": g_attentions,
                "f_attentions": f_attentions,
                "o_attentions": o_attentions,
            }
            torch.save(all_attentions, "case_attentions_cpu.pkl")
        convert_cuda_into_cpus(all_attentions)

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log()
            self.resume_log()
            if self.config.model in ['gpt2']:
                self.set_AR_FFT()
            else:
                # self.set_FFT()
                self.set_PEFT()
            # self.train()
            self.joint_training()
            self.load_state(self.config.dataset)
            with torch.no_grad():
                if self.config.dataset in ['mtl']:
                    self.eval_domain(self.test_itr)
                test_metrics = self.eval(self.test_itr, global_version=False)
                print()
                test_logs = self.get_logging(test_metrics, eval="Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

                test_metrics = self.eval(self.test_itr, global_version=True)
                print()
                test_logs = self.get_logging(test_metrics, eval="Global Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

            # for continue learning
            dev_metrics_general = self.eval(self.dev_itr, global_version=True)
            self.best_dev_acc_general = dev_metrics_general['accuracy']

            self.continue_learning()
            self.load_state(self.config.dataset)
            with torch.no_grad():
                if self.config.dataset in ['mtl']:
                    self.eval_domain(self.test_itr)
                test_metrics = self.eval(self.test_itr, global_version=False)
                print()
                test_logs = self.get_logging(test_metrics, eval="Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

                test_metrics = self.eval(self.test_itr, global_version=True)
                print()
                test_logs = self.get_logging(test_metrics, eval="Global Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)
        elif run_mode == 'val':
            self.load_state(self.config.dataset)
            with torch.no_grad():
                if self.config.dataset in ['mtl']:
                    self.eval_domain(self.dev_itr)
                test_metrics = self.eval(self.dev_itr)
                print(test_metrics)
        elif run_mode == 'test':
            self.load_state(self.config.dataset)
            with torch.no_grad():
                if self.config.dataset in ['mtl']:
                    self.eval_domain(self.test_itr)
                test_metrics = self.eval(self.test_itr, global_version=False)
                test_logs = self.get_logging(test_metrics, eval="Testing")
                print(test_logs)

                test_metrics = self.eval(self.test_itr, global_version=True)
                print()
                test_logs = self.get_logging(test_metrics, eval="Global Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)
        elif run_mode == 'g_test':
            self.load_state(self.config.dataset)
            with torch.no_grad():
                # original settings
                test_metrics = self.eval(self.test_itr, global_version=False)
                print()
                test_logs = self.get_logging(test_metrics, eval="Original Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

                test_metrics = self.eval(self.test_itr, global_version=True)
                print()
                test_logs = self.get_logging(test_metrics, eval="Global Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

                test_metrics_random = self.random_eval(self.test_itr)
                print()
                test_logs = self.get_logging(test_metrics_random, eval="Random Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

                # setting average for general view embedding
                self.net.user_embedding.weight.data = torch.mean(self.net.user_embedding.weight.data, dim=0, keepdim=True).repeat(self.config.usr_size, 1)
                self.net.item_embedding.weight.data = torch.mean(self.net.item_embedding.weight.data, dim=0, keepdim=True).repeat(self.config.prd_size, 1)
                test_metrics_avg = self.eval(self.test_itr, global_version=False)
                print()
                test_logs = self.get_logging(test_metrics_avg, eval="Avg Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)

                # setting zero for general view embedding
                self.net.user_embedding.weight.data = torch.zeros_like(self.net.user_embedding.weight.data,
                                                                       device=self.config.device,
                                                                       dtype=self.net.user_embedding.weight.data.dtype)
                self.net.user_embedding.weight.data = torch.zeros_like(self.net.user_embedding.weight.data,
                                                                       device=self.config.device,
                                                                       dtype=self.net.user_embedding.weight.data.dtype
                                                                       )
                test_metrics_zero = self.eval(self.test_itr, global_version=False)
                print()
                test_logs = self.get_logging(test_metrics_zero, eval="Zero Testing")
                print("\r" + test_logs)
                self.logging(self.log_path, test_logs)
        elif run_mode == 'case':
            self.load_state(self.config.dataset)
            model_name_or_path = MODEL_MAP[self.config.model]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            with torch.no_grad():
                # self.finding_case(self.test_itr)
                self.case_2()


        else:
            exit(-1)