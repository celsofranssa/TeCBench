import torch
from pytorch_lightning import LightningModule


class AttentivePooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self, hparms):
        super(AttentivePooling, self).__init__()
        self.att_pooling = torch.nn.Sequential(
            torch.nn.Linear(hparms.hidden_size, hparms.max_length),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, attention_mask, encoder_outputs):
        """
        """
        pooler_output, hidden_states = encoder_outputs.pooler_output, encoder_outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        att_weight = self.att_pooling(pooler_output).unsqueeze(-1)

        # put 0 over PADs
        hidden_states = hidden_states * attention_mask

        return (att_weight * hidden_states).sum(axis=1)
