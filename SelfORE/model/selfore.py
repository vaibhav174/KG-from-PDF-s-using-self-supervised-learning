from model.classifier import Classifier
from sklearn.cluster import KMeans
from model.adaptive_clustering import AdaptiveClustering
from tqdm import tqdm
import torch


class SelfORE:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.k = config['k']
        self.loop_num = config['loop']
        cluster = config['cluster']
        self.model_path = config["model_path"]
        self.pretrain_ac = config["pretrain_ac"]

        # initialize classifier
        self.classifier = Classifier(
            k=self.k,
            sentence_path=config['sentence_path'],
            max_len=config['bert_max_len'],
            batch_size=config['batch_size'],
            epoch=config['epoch']
        )

        # get size of hidden state of classifier
        hidden_size = self.classifier.get_hidden_state_size()

        # initialize adaptive clustering
        if cluster == 'kmeans':
            self.pseudo_model = KMeans(n_clusters=self.k, random_state=0)
        
        elif cluster == 'adpative_clustering':
            self.pseudo_model = AdaptiveClustering(n_clusters=self.k, input_dim=hidden_size)
            # if self.pretrain_ac is true, pretrain the encoder with an auto-encoder
            if self.pretrain_ac:
                bert_embs = lambda: self.classifier.get_hidden_state()
                self.pseudo_model.pretrain_enc(bert_embs, batch_size=config['batch_size'])
        
        else:
            raise Exception(f'Clustering algorithm {cluster} not support yet')


    def loop(self):
        # training loop
        print("=== Generating Pseudo Labels...")
        bert_embs = lambda: self.classifier.get_hidden_state()
        pseudo_labels = self.pseudo_model.fit(bert_embs, batch_size=self.batch_size).labels_
        print("=== Generating Pseudo Labels Done")

        print("=== Training Classifier Model...")
        self.classifier.train(pseudo_labels)
        print("=== Training Classifier Model Done")

    def start(self):
        print("starting ...")
        for _ in tqdm(range(self.loop_num)):
            self.loop()

    def save_model(self):
        torch.save(self.classifier.state_dict(), self.model_path)

    def read_model(self):
        self.classifier.read_model(self.model_path)
