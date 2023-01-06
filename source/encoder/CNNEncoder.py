import torch.nn
from pytorch_lightning import LightningModule
from torch import nn


class CNNEncoder(LightningModule):
    """Represents a text as a dense vector."""

    def __init__(self, vocabulary_size, hidden_size, max_length):
        super(CNNEncoder, self).__init__()
        self.l1 = nn.Embedding(vocabulary_size, hidden_size)
        self.l2 = nn.Dropout(0.8)
        self.l3 = nn.Conv1d(in_channels=max_length, out_channels=max_length, kernel_size=256)
        self.l4 = nn.ReLU()
        self.l5 = nn.MaxPool1d(kernel_size=2)
        self.l6 = nn.Dropout(0.6)
        self.l7 = nn.Linear(256, 512)
        self.l8 = nn.ReLU()
        self.l9 = nn.Dropout(0.2)

        # model_en = tf.keras.Sequential([
        #     tf.keras.Input(shape=(1,), dtype=tf.string),
        #     vectorize_layer_en,
        #     layers.Embedding(max_features + 1, embedding_dim),
        #     layers.Dropout(0.8),
        #
        #     layers.Conv1D(256, 16, activation='relu'),
        #     layers.MaxPooling1D(),
        #     layers.Dropout(0.6),
        #
        #     layers.Dense(512, activation='relu'),
        #
        #     layers.GlobalAveragePooling1D(),
        #     layers.Dropout(0.2),
        #     layers.Dense(10)
        # ])

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(torch.mean(x, dim=1))
        x = self.l8(x)
        return self.l9(x)






