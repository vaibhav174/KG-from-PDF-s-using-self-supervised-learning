'''
Code from: https://github.com/THU-BPM/SelfORE
'''

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import math
from sklearn.cluster import KMeans


def buildNetwork(layers, activation="relu", dropout=0):
    # helper function to assemble MLP with layer sizes defined in the layers list
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class AdaptiveClustering(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], activation="relu", dropout=0, alpha=1.):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        net_structure = [input_dim] + encodeLayer
        rev_net_structure = [z_dim] + list(reversed(net_structure))

        # build encoder & decoder network
        self.encoder = buildNetwork(
            net_structure, activation=activation, dropout=dropout)
        self.decoder = buildNetwork(
            rev_net_structure, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        # output layer
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.labels_ = []

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        # load model from a saved state dict
        pretrained_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        # forward pass
        h = self.encoder(x)
        z = self._enc_mu(h)

        # compute q -> NxK
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

    def forward_autoenc(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h_pred = self.decoder(z)

        return h_pred

    def encodeBatch(self, dataloader, islabel=False):
        # calculates pseudo labels for a dataloader
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        ylabels = []
        self.eval()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # forward all inputs through the adaptive clustering network
            inputs = Variable(inputs)
            z, _ = self.forward(inputs)
            encoded.append(z.data.cpu())
            ylabels.append(labels)

        # concatenate tensors
        encoded = torch.cat(encoded, dim=0)
        ylabels = torch.cat(ylabels)

        # if labels should be returned, add them to out, else only return encodings
        if islabel:
            out = (encoded, ylabels)
        else:
            out = encoded
        return out

    def loss_function(self, p, q):
        # KL-divergence implementation
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        # auxilliary distribution
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, x_generator, y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-4):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        # calculate KMeans cluster centroids and preedictions
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = torch.Tensor()
        for xbatch in x_generator():
            xbatch = torch.Tensor(xbatch)
            data_temp, _ = self.forward(xbatch)
            data = torch.concat((data, data_temp))
        y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        y_pred_last = y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        # training loop
        self.train()
        
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                q = torch.Tensor()
                for xbatch in x_generator():
                    xbatch = torch.Tensor(xbatch)
                    _, q_temp = self.forward(xbatch)
                    q = torch.concat((q, q_temp))
                p = self.target_distribution(q).data

                # evaluate the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / data.shape[0]
                y_pred_last = y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # train 1 epoch
            train_loss = 0.0
            num = data.shape[0]
            for batch_idx, xbatch in enumerate(x_generator()):
                xbatch = torch.Tensor(xbatch)
                pbatch = p[batch_idx * batch_size: min((batch_idx+1)*batch_size, num)]

                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                # forward & backward pass
                z, qbatch = self.forward(inputs)
                loss = self.loss_function(target, qbatch)
                train_loss += loss.data*len(inputs)
                loss.backward()
                optimizer.step()

            print("#Epoch %3d: Loss: %.4f" % (
                epoch+1, train_loss / num))

        # get labels from the last predictions
        self.labels_ = np.array(y_pred_last)
        return self

    def pretrain_enc(self, x_generator, lr=0.001, num_batches=2, batch_size=256, num_epochs=2):
        # pretraining for the encoder
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.MSELoss()

        self.train()

        print(f"Starting pre-training the encoder...")
        for epoch in range(num_epochs):
            train_loss = 0.0
            num = 0
            i = 0
            for xbatch in x_generator():
                optimizer.zero_grad()
                xbatch = torch.Tensor(xbatch)
                inputs = Variable(xbatch)
                outputs = self.forward_autoenc(inputs)

                loss = criterion(inputs, outputs)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num += xbatch.shape[0]
                i += 1
                if i > num_batches:
                    break

            print("#Epoch %3d: Loss: %.4f" % (
                epoch+1, train_loss / num))
