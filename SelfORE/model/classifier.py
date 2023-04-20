'''
Code from: https://github.com/THU-BPM/SelfORE
'''

import random
import json

import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from torch.optim import AdamW
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score
import numpy as np
import bcubed
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, k, sentence_path, max_len, batch_size, epoch, rel_available=False):
        """
        Loads a file at sentence_path and prepares all variables and data.
        File needs to be in JSON format, each item in the list should be a dict. The sentence has to be in the 'text' key, and if 
            tag_available=True, then there should be a 'relation' key.
        """
        with open(sentence_path) as f:
            data = json.load(f)
            self.sentences = [d["text"] for d in data]
            if rel_available:
                self.relations = np.array([d["relation"] for d in data])

        self.rel_available = rel_available
        self.k = k
        self.epoch = epoch
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.device = self.get_device()
        self.model = self.model.to(self.device)
        self.input_ids, self.attention_masks = self.prepare_data()

        # variable to check if network is trained or not, needed for later checks
        self.trained = False

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    @staticmethod
    def get_tokenizer():
        # initialize tokenizer
        # MobileBertTokenizer is identical to BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        special_tokens_dict = {'additional_special_tokens': [
            '[E1]', '[E2]', '[/E1]', '[/E2]']}  # add special token
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    def get_model(self):
        # initialize transformer network from Huggingface
        # model = BertForSequenceClassification.from_pretrained(
        #     "bert-base-uncased",
        #     num_labels=self.k,
        #     output_attentions=False,
        #     output_hidden_states=True,
        # )
        model = MobileBertForSequenceClassification.from_pretrained(
            "lordtt13/emo-mobilebert",
            num_labels=self.k,
            output_attentions=False,
            output_hidden_states=True,
            ignore_mismatched_sizes=True,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def prepare_data(self):
        # load data
        input_ids = []
        attention_masks = []
        for sent in self.sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,                        # Sentence to encode.
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                max_length=self.max_len,     # Pad & truncate all sentences.
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',         # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        print('Original: ', self.sentences[0])
        print('Token IDs:', input_ids[0])
        return input_ids, attention_masks

    def get_entity_idx(self):
        e1_tks_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        e2_tks_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        entity_idx = []
        for input_id in self.input_ids:
            e1_idx = (input_id == e1_tks_id).nonzero().flatten().tolist()[0]
            e2_idx = (input_id == e2_tks_id).nonzero().flatten().tolist()[0]
            entity_idx.append((e1_idx, e2_idx))
        entity_idx = torch.Tensor(entity_idx)
        return entity_idx

    def get_hidden_state(self):
        dataset = TensorDataset(self.input_ids, self.attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # calculate one batch at a time and wait for the next call of the loop to calculate the next one
        # Solves memory issues
        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            out = self.model(b_input_ids, b_input_mask)[1][-1].to('cpu').detach().numpy().flatten().reshape(len(b_input_ids), -1)
            yield out

    def get_hidden_state_size(self):
        dataset = TensorDataset(self.input_ids, self.attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # perform one batch calculation in order to get the size of the hidden state
        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            _, shape_1, shape_2 = self.model(b_input_ids, b_input_mask)[1][0].shape
            break

        return shape_1 * shape_2

    def train(self, labels):
        # define labelss and data
        labels = torch.tensor(labels).long()
        dataset = TensorDataset(self.input_ids, self.attention_masks, labels)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size
        )

        self.validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size
        )

        # define training variables
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        epochs = self.epoch
        total_steps = len(self.train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # set seeds for reproducability, remove in proper training
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        if torch.cuda.is_available():
            self.model.cuda()

        # run epochs
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            self.train_epoch()

        print("Training done!")
        self.trained = True

    def train_epoch(self):
        # training loop for one epoch
        total_train_loss = 0
        self.model.train()
        for batch in self.train_dataloader:
            # get batch data
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.model.zero_grad()
            
            # perform forward & backward pass 
            output = self.model(b_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_input_mask,
                                         labels=b_labels)

            loss = output.loss
            logits = output.logits
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        avg_train_loss = total_train_loss / len(self.train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("")
        print("Running Validation...")
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        # validation loop
        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            with torch.no_grad():
                output = self.model(b_input_ids,
                                               token_type_ids=None,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
            loss = output.loss
            logits = output.logits
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    
    def test(self, model_path=None, return_idx=True, verbose=True):
        # check if model was trained, if not load model
        if not self.trained:
            self.read_model(model_path)
            self.trained = True

        self.model.eval()

        dataset = TensorDataset(self.input_ids, self.attention_masks)

        test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size
        )

        logits = torch.tensor([], dtype=torch.float32)

        print("Starting testing!")
        for i, batch in enumerate(tqdm.tqdm(test_dataloader)):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            logits = torch.concat((logits, self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)['logits'].detach().to('cpu')), 0)

        max_idx = logits.max(axis=1).indices
        cluster_idx = [int(i) for i in max_idx.unique()]
        cluster_count = {int(i): torch.where(max_idx == i)[0].shape[0] for i in cluster_idx}

        # If verbose is true, create output text with cluster examples
        if verbose:
            for i in cluster_idx:
                print(f"cluster {i} has {cluster_count[i]} occurence(s)")
                print(f"Example sentences:")
                try:
                    # try to get 5 cluster examples, if there are fewer than 5 take all
                    sample_idx = random.sample(torch.where(max_idx == i)[0].tolist(), 5)
                except ValueError:
                    sample_idx = torch.where(max_idx == i)[0]
                for j in sample_idx:
                    print(f"\t{self.sentences[j]}")

        # barplot with cluster distribution
        plt.bar(cluster_idx, cluster_count.values())
        plt.show()
        plt.savefig(f"data/plots/output.png", dpi=200)

                # If ground-truth relations are available, evaluate metrics for the predicted relations
        if self.rel_available:
            # Assign index to each relation
            rel_dict = {rel: i for i, rel in enumerate(np.unique(self.relations))}

            # TODO: if n-grams were calculated, compare them to the ground truth relations

            # TODO: assign the majority label to each cluster
            cluster_labels = {}
            for i in cluster_idx:
                # get the counts of each label in the cluster i
                cluster_occ = np.where(max_idx == i)[0]
                cluster_relations = self.relations[cluster_occ]
                unique, pos = np.unique(cluster_relations, return_inverse=True)
                counts = np.bincount(pos)

                # calculate majority label
                maj_label = unique[counts.argmax()]
                cluster_labels[i] = maj_label

            # Calculate different metrics
            gold_dict = {i : set([self.relations[i]]) for i in range(len(self.relations))}
            gold_dict_idx = {i: rel_dict[self.relations[i]] for i in range(len(self.relations))}
            pred_dict = {i: set([cluster_labels[int(max_idx[i])]]) for i in range(len(max_idx))}
            pred_dict_idx = {i: rel_dict[cluster_labels[int(max_idx[i])]] for i in range(len(max_idx))}
            
            # TODO: calculate V-measure, B^3 and ARI measures
            # Calculate B^3
            # B^3 metric expl.: https://brenocon.com/blog/2013/08/probabilistic-interpretation-of-the-b3-coreference-resolution-metric/
            # TODO: sainity check for B^3 measure
            if len(gold_dict) != len(pred_dict):
                raise ValueError("dictionaries with ground-truth and predicted relations are not the same length!")

            b3_prec = bcubed.precision(pred_dict, gold_dict)
            b3_recall = bcubed.recall(pred_dict, gold_dict)
            b3_f1 = bcubed.fscore(b3_prec, b3_recall)

            # Calculate V-measure
            # V-measure expl.: https://towardsdatascience.com/v-measure-an-homogeneous-and-complete-clustering-ab5b1823d0ad
            v_homogeneity, v_completeness, v_f1 = homogeneity_completeness_v_measure(list(gold_dict_idx.values()), list(pred_dict_idx.values()))

            # Calculate ARI
            ari = adjusted_rand_score(list(gold_dict_idx.values()), list(pred_dict_idx.values()))

            print("--- Test measures ---")
            print(f"B-Cubed:\t{b3_f1} (F1)\t{b3_prec} (Prec.)\t{b3_recall} (Recall)")
            print(f"V-Measure:\t{v_f1} (F1)\t{v_homogeneity} (Hom.)\t{v_completeness} (Comp.)")
            print(f"ARI:\t\t{ari}")

        if return_idx:
            return max_idx

    def inference(self, model_path=None):
        # function for forward pass, used in testing
        # check if model was trained, if not load model
        if not self.trained:
            self.read_model(model_path)
            self.trained = True

        self.model.eval()

        # load data
        dataset = TensorDataset(self.input_ids, self.attention_masks)
        test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size
        )
        logits = torch.tensor([], dtype=torch.float32)

        print("Starting testing!")
        for i, batch in enumerate(tqdm.tqdm(test_dataloader)):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            logits = torch.concat((logits, self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)['logits'].detach().to('cpu')), 0)

        # assign samples to their max probability cluster
        max_idx = logits.max(axis=1).indices

        return max_idx

    def state_dict(self):
        # get the state dict of the model
        return self.model.state_dict()
    
    def read_model(self, path):
        # load the model from a state dict
        return self.model.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
