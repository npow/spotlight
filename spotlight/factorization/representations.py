"""
Classes defining user and item latent representations in
factorization models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class HybridContainer(nn.Module):

    def __init__(self,
                 latent_module,
                 user_module=None,
                 context_module=None,
                 item_module=None,
                 mu=None):

        super(HybridContainer, self).__init__()

        self.latent = latent_module
        self.user = user_module
        self.context = context_module
        self.item = item_module
        self.mu = mu

    def forward(self, user_ids,
                item_ids,
                user_features=None,
                context_features=None,
                item_features=None):

        user_representation, user_bias = self.latent.user_representation(user_ids)
        item_representation, item_bias = self.latent.item_representation(item_ids)

        if self.user is not None:
            user_representation += self.user(user_features)
        if self.context is not None:
            user_representation += self.context(context_features)
        if self.item is not None:
            feature_representation, feature_bias = self.item(item_features)
            item_representation += feature_representation
            item_bias += feature_bias

        #dot = F.cosine_similarity(user_representation, item_representation).unsqueeze(1)
        dot = (user_representation * item_representation).sum(dim=1, keepdims=True)
        logits = torch.sigmoid(dot + user_bias + item_bias)
        return logits

        return dot + user_bias + item_bias + self.mu


class FeatureNet(nn.Module):

    def __init__(self, num_features, embedding_dim, nonlinearity='linear', sparse=False):

        super(FeatureNet, self).__init__()

        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = F.sigmoid
        elif nonlinearity == 'linear':
            self.nonlinearity = nn.Identity()
        else:
            raise ValueError('Nonlineariy must be one of '
                             '(tanh, relu, sigmoid, linear)')

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(num_features, embedding_dim, sparse=sparse, padding_idx=0)
        self.embeddings.weight.data.normal_(0, 0.1)
        self.biases = ZeroEmbedding(num_features, 1, sparse=sparse)

    def forward(self, features):
        feature_embedding = self.embeddings(features).sum(dim=1)
        feature_bias = self.biases(features).sum(dim=1)
        return self.nonlinearity(feature_embedding), feature_bias


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, embedding_dim=32, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

        self.user_embeddings.weight.data.normal_(0, 0.1)
        self.item_embeddings.weight.data.normal_(0, 0.1)

    def user_representation(self, user_ids):

        user_embedding = self.user_embeddings(user_ids)
        user_embedding = user_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)

        return user_embedding, user_bias

    def item_representation(self, item_ids):

        item_embedding = self.item_embeddings(item_ids)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        item_bias = self.item_biases(item_ids).view(-1, 1)

        return item_embedding, item_bias

    def forward(self, user_representation, user_bias, item_representation, item_bias):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        dot = F.cosine_similarity(user_representation, item_representation).unsqueeze(1)

        return dot + user_bias + item_bias



class HybridNCF(nn.Module):

    def __init__(self, latent_module, user_module=None, context_module=None, item_module=None, layers=[16, 8], dropout=0.0):
        super().__init__()
        assert False
        embedding_dim = latent_module.embedding_dim
        layers = [embedding_dim*2, embedding_dim]
        assert (layers[0] == 2 * embedding_dim), "layers[0] must be 2*embedding_dim"
        self.latent = latent_module
        self.user = user_module
        self.context = context_module
        self.item = item_module
        self.dropout = dropout

        self.fc_layers = nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        self.output_layer = nn.Linear(layers[-1], 1)


    def forward(self, user_ids,
                item_ids,
                user_features=None,
                context_features=None,
                item_features=None):
        user_embedding, user_bias = self.latent.user_representation(user_ids)
        item_embedding, item_bias = self.latent.item_representation(item_ids)

        if self.user is not None:
            user_embedding += self.user(user_features)
        if self.context is not None:
            user_embedding += self.context(context_features)
        if self.item is not None:
            item_embedding += self.item(item_features)

        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.output_layer(x)
        return logits
