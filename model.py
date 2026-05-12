import torch
import torch.nn as nn

# extend from nn.Module
class DeepFM(nn.Module):
    def __init__(
            self,
            num_features,
            field_size,
            embedding_dim = 8
    ):

        super().__init__()

        # --------------------------------
        # Linear Part
        # --------------------------------
        # vector
        self.linear = nn.Embedding(
            num_features,
            1
        )

        # --------------------------------
        # Shared Embedding
        # --------------------------------
        # embedding part
        # put discrete  features/ids into embedding layer
        # nn is good at handling the continous vector space
        # not good at handling the discrete  features/ids

        # ont hot enconding is very sparse   [0 0 0 1 0 0 0 0 0 0]  high dimension
        # embedding is dense and low dimension, and can capture the sematic relations
        # [num_features, embedding_dim]
        self.embedding = nn.Embedding(
            num_features,
            embedding_dim
        )

        # --------------------------------
        # Deep Neural Network
        # --------------------------------
        # multi-layer perception
        self.mlp = nn.Sequential(
            # linear layer
            # input     [batch, field_size * embedding_dim]
            # output    [batch, 128]
            # weight    [field_size * embedding_dim, 128]
            # bias      [128]
            # just extand the feature space to 128, information is not lost or increased, just the same information in a different space
            nn.Linear(field_size * embedding_dim, 128),

            # activivation function
            # increase the non-linearity
            # if there is no activation function,
            # the whole model is just a linear model, no matter how many layers you have, the output is just a linear combination of the input
            # activation function can let the model learn more
            # complex functions, feature interaction and non-linear relationship
            # feature interaction is very complex and linear model cannt capture it
            nn.ReLU(),

            # dropout layer
            # randomly set some of neurons to zero during training
            # to avoid overfitting
            # to avoid the model just rely on some of the neurons
            # it should learn to use all the neurons
            # overfitting: models learn the details of the training data and noies,
            # the model will perform worse on the new data, becuase it just memorize the trainning data
            nn.Dropout(0.2),

            # linear layer
            # compress the feature space to 64
            nn.Linear(128, 64),

            # activivation function
            nn.ReLU(),

            # dropout layer
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, field_size]

        # linear part
        # add for all embedding dimensions
        linear_part = self.linear(x).sum(dim = 1)

        # embedding
        # shape: [batch, field_size, embedding_dim]
        embeddings = self.embedding(x)

        # FM part
        # add at dimension 1, remove the filed_size dimension
        # first sum then square
        # x1^2 + x2^2 + x3^2 + ... + xn^2 + 2*sum(xi * xj)
        square_of_sum = embeddings.sum(dim = 1) ** 2

        # first square then sum
        # x1^2 + x2^2 + x3^2 + ... + xn^2   -> independent punishment
        # easy, independent, grad is easy to compute, good to optimize
        sum_of_square = (embeddings ** 2).sum(dim = 1)

        # factorization machine part, add the interaction between features,
        # 0.5 just remove the extra 2
        # standard FM form is sum(xi * xj)
        # directly compute the interaction without computing the pairwise
        # save the computational cost
        fm_part = 0.5 * (
            # # extra 2 * sum(xi * xj)   -> interaction between features
            square_of_sum - sum_of_square
        ).sum(dim = 1, keepdim = True)  # sum at the dim 1,  -> [batch, 1]

        # deep part
        # reshape and flattern: [batch, field_size * embedding_dim]
        deep_input = embeddings.view(
            embeddings.size(0),   # batch size
            -1                    # calculate by itself
        )
            # use the dnn to learn
        deep_part = self.mlp(deep_input)

        # output
                # linear : first order direclty add weight of each feature- independent influence
                # fm: second order feature interaction, add the interaction multiplication of each feature
                # deep: high order feature non-linear interaction
        output = linear_part + fm_part + deep_part

        # change to the pobabibility between 0 and 1
        # squeeze: remove the dimension with size 1, [batch, 1] -> [batch]
        return torch.sigmoid(
            output.squeeze()
        )
