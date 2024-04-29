import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.randn(num_dir * 2, x.size(0), hidden_size)).to(device)
    return Variable(torch.randn(num_dir * 2, x.size(0), hidden_size)).to(
        device)  # num_layers * num_directions,max_batch_size, real_hidden_size


"""
Encoder
"""


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size_enc: int, num_layers_enc: int, seq_len: int):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size_enc
        self.num_layers = num_layers_enc
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, input_data: torch.Tensor):
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))

        input_encoded = Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size))

        # for t in range(self.seq_len):
        #     print('input_data size:{},details:{}'.format(input_data.size(), input_data))
        #     output, (h_t, c_t) = self.lstm(input_data[:, t, :].unsqueeze(0), (h_t, c_t))
        #     input_encoded[:, t, :] = output[:, t, :]  # last layer h_t

        output, (h_t, c_t) = self.lstm(input_data, (h_t, c_t))
        input_encoded = output  # last layer h_t

        return output, input_encoded


class Decoder(nn.Module):
    def __init__(self, hidden_size_enc: int, hidden_size_dec: int, num_layers_dec: int, seq_len: int,
                 output_size: int):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.encoder_hidden_size = hidden_size_enc
        self.decoder_hidden_size = hidden_size_dec
        self.out_feats = output_size
        self.num_layers = num_layers_dec

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats, hidden_size=self.decoder_hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.encoder_hidden_size + self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input,  [batch_size, seq_len, hidden_size]
            y_history: (torch.Tensor): shifted target
        """
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size), init_hidden(input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded.to(device)), dim=2)

            x = F.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded.to(device))[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        return self.fc_out(torch.cat((h_t[0], context.to(device)), dim=1))  # predicting value at t=self.seq_length+1


class ALSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size_enc, num_layers_enc, seq_len, hidden_size_dec, num_layers_dec,
                 output_size):
        """
        Initialize the network.

        """
        super(ALSeq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size_enc, num_layers_enc, seq_len).to(device)
        self.decoder = Decoder(hidden_size_enc, hidden_size_dec, num_layers_dec, seq_len, output_size).to(device)

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
        """

        _, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output, y_hist.float())

        return outputs
