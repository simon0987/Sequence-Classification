import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    """
    A `Highway layer (https://arxiv.org/abs/1505.00387)` does a gated combination of its
    input and a non-linear transformation of its input. `y = g * H(x) + (1 - g) * x`,
    where H is a linear transformation followed by an element-wise non-linearity, and g
    is an element-wise gate, computed as `sigmoid(T(x))`.

    This module will apply a fixed number of highway layers to its input, returning the
    final result.

    References:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py
    https://github.com/allenai/bilm-tf/blob/master/bilm/model.py

    Parameters
    ----------
    input_size : ``int``
        The dimensionality of `x`.  We assume the input has shape.
    n_layers : ``int``
        The number of highway layers to apply to the input.
    """
    def __init__(self, input_size, n_layers):
        super().__init__()

        self.Ts = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(n_layers)])
        self.Hs = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(n_layers)])

        # Initialize the bias of the gating function T(x) to negative values so that
        # the network is initially biased towards carry behavior.
        # See the paper for more details.
        for T in self.Ts:
            for name, param in T.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0, std=np.sqrt(1 / input_size))
                elif 'bias' in name:
                    nn.init.constant_(param, -2)

        for H in self.Hs:
            for name, param in H.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0, std=np.sqrt(1 / input_size))
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x):
        for T, H in zip(self.Ts, self.Hs):
            g = torch.sigmoid(T(x))
            y = F.relu(H(x))
            x = g * y + (1 - g) * x

        return x


class CharEmbedding(nn.Module):
    """
    This module compute the character embedding of the input. Multiple highway networks
    and a linear transformation are applied to the CNN outputs.

    References:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py
    https://github.com/allenai/bilm-tf/blob/master/bilm/model.py

    Parameters
    ----------
    num_embeddings : ``int``
        Size of the character vocabs.
    embedding_dim : ``int``
        Size of the embedding vector.
    padding_idx : ``int``
        Index of the padding in character vocabs.
    conv_filters : ``List[Tuple(int, int)]
        Each tuple (kernel_size, num_filters) in the list defines a CNN.
    n_highways : ``int``
        Number of highway layers.
    projection_size : ``int``
        Size of the final output.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx, conv_filters,
                 n_highways, projection_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, n_filters, kernel_size)
            for kernel_size, n_filters in conv_filters])
        n_filters = sum([f[1] for f in conv_filters])
        self.highway = HighwayNetwork(n_filters, n_highways)
        self.projection = nn.Linear(n_filters, projection_size)

        for param in self.embedding.parameters():
            nn.init.uniform_(param, a=-1, b=1)

        for conv in self.convs:
            for name, param in conv.named_parameters():
                if 'weight' in name:
                    nn.init.uniform_(param, a=-0.05, b=0.05)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

        for name, param in self.projection.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=np.sqrt(1 / n_filters))
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Parameters
        ----------
        x : ``torch.tensor(shape=(batch_size, sentence_len, word_len), dtype=torch.int64)``
            Using an example to illustrate the input will be clearer.
            Assuming the raw input contains two sentences:
                '<sos> Hello world !'
                '<sos> Nice to meet you .'
            The sentences should be tokenized into:
                [['<sos>', 'Hello', 'world', '!'],
                 ['<sos>', 'Nice', 'to', 'meet', 'you', '.']]
            Then split into characters:
                [[['<sos>'],
                  ['H', 'e', 'l', 'l', 'o'],
                  ['w', 'o', 'r', 'l', 'd'],
                  ['!']],
                 [['<sos>'],
                  ['N', 'i', 'c', 'e'],
                  ['t', 'o'],
                  ['m', 'e', 'e', 't'],
                  ['y', 'o', 'u'],
                  ['.']]]
            Then add padding (and truncate) so that each word and each sentence has the
            same length:
                [[['<sos>', '<pad>', '<pad>', '<pad>', '<pad>'],
                  ['H',     'e',     'l',     'l',     'o'],
                  ['w',     'o',     'r',     'l',     'd'],
                  ['!',     '<pad>', '<pad>', '<pad>', '<pad>']],
                  ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>']],
                  ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>']],
                 [['<sos>', '<pad>', '<pad>', '<pad>', '<pad>'],
                  ['N',     'i',     'c',     'e',     '<pad>'],
                  ['t',     'o',     '<pad>', '<pad>', '<pad>'],
                  ['m',     'e',     'e',     't',     '<pad>'],
                  ['y',     'o',     'u',     '<pad>', '<pad>'],
                  ['.',     '<pad>', '<pad>', '<pad>', '<pad>']]]
            Finally, transform the characters into their corresponding index:
            (I used the ascii number as the index in this example, and assign 256, 257
            to <pad>, <sos>, respectively.)
                [[[257, 256, 256, 256, 256],
                  [ 72, 101, 108, 108, 111],
                  [119, 111, 114, 108, 100],
                  [ 33, 256, 256, 256, 256]],
                  [256, 256, 256, 256, 256]],
                  [256, 256, 256, 256, 256]],
                 [[257, 256, 256, 256, 256],
                  [ 78, 105,  99, 101, 256],
                  [116, 111, 256, 256, 256],
                  [109, 101, 101, 116, 256],
                  [121, 111, 117, 256, 256],
                  [ 46, 256, 256, 256, 256]]]
            In this case, the input `x` is a tensor of shape `(2, 6, 5)`.

        Returns
        -------
        A torch.tensor of shape `(batch_size, sentence_len, projection_size)` and dtype
        `torch.float32`.
        """
        emb = self.embedding(x)
        batch_size, seq_len, word_len, emb_dim = emb.shape
        emb = emb.transpose(2, 3).reshape(-1, emb_dim, word_len)
        embs = []
        for conv in self.convs:
            _emb = conv(emb).max(dim=-1)[0]
            _emb = F.relu(_emb)
            embs.append(_emb)
        emb = torch.cat(embs, dim=-1).reshape(batch_size, seq_len, -1)
        emb = self.highway(emb)
        emb = self.projection(emb)

        return emb

class ELMO(torch.nn.Module):
	def __init__(self, word_vocab, char_vocab):
		super(ELMO,self).__init__()
		self.embedding = CharEmbedding(
			num_embeddings = char_vocab.__len__(),
			embedding_dim = 16,
			padding_idx = char_vocab.sp.pad.idx,
			conv_filters = [(1, 32), (2, 64), (3, 128), (4, 128), (5, 256), (6, 256), (7, 512)],
			n_highways = 2,
			projection_size = 512
			)
		self.lstm_forward = nn.LSTM(
				input_size = 512,
				num_layers = 1,
				hidden_size = 2048,
				bidirectional = False,
				batch_first=True
			)
		self.linear_forward = nn.Linear(2048,512)
		self.lstm_forward2 = nn.LSTM(
				input_size = 512,
				num_layers = 1,
				hidden_size = 2048,
				bidirectional = False,
				batch_first=True
			)
		self.linear_forward2 = nn.Linear(2048,512)
		self.lstm_backward = nn.LSTM(
				input_size = 512,
				num_layers = 1,
				hidden_size = 2048,
				bidirectional = False,
				batch_first=True
			)
		self.linear_backward = nn.Linear(2048,512)
		self.lstm_backward2 = nn.LSTM(
				input_size = 512,
				num_layers = 1,
				hidden_size = 2048,
				bidirectional = False,
				batch_first=True
			)
		self.linear_backward2 = nn.Linear(2048,512)
		self.output_layer = nn.AdaptiveLogSoftmaxWithLoss(
				in_features = 512,
				n_classes = word_vocab.__len__(),
				cutoffs = [100,1000,10000]
			)
	def forward(self,forward,backward):
		forward_x = self.embedding(forward)
		#(batch_size, sentence_len, projection_size) (32,63,512)
		forward_x = self.lstm_forward(forward_x)[0]
		forward_x = self.linear_forward(forward_x)
		forward_x = self.lstm_forward2(forward_x)[0]
		forward_x = self.linear_forward2(forward_x)
		#(32,63,512)
		forward_output = self.output_layer.log_prob(forward_x.view(-1,512))
		# forward_x.view(-1,512) : (2016,512)
		# forward_output : (2016,79899)
		backward_x = self.embedding(backward)
		backward_x = self.lstm_backward(backward_x)[0]
		backward_x = self.linear_backward(backward_x)
		backward_x = self.lstm_backward2(backward_x)[0]
		backward_x = self.linear_backward2(backward_x)
		backward_output = self.output_layer.log_prob(backward_x.view(-1,512))

		output = torch.cat((forward_output,backward_output),0)
		return output

	def get_embedding(self,forward,backward):
		char_embedding_forward = self.embedding(forward)
		forward_x = self.lstm_forward(char_embedding_forward)[0]
		lstm_forward_first = self.linear_forward(forward_x)
		forward_x = self.lstm_forward2(lstm_forward_first)[0]
		lstm_forward_second = self.linear_forward2(forward_x)

		char_embedding_backward = self.embedding(backward)
		backward_x = self.lstm_backward(char_embedding_backward)[0]
		lstm_backward_first = self.linear_backward(backward_x)
		backward_x = self.lstm_backward2(lstm_backward_first)[0]
		lstm_backward_second = self.linear_backward2(backward_x)

		char_embedding = torch.cat((char_embedding_forward,char_embedding_backward),2)
		lstm_forward_out = torch.cat((lstm_forward_first,lstm_backward_first),2)
		lstm_backward_out = torch.cat((lstm_forward_second,lstm_forward_second),2)
		result = []
		result.append(char_embedding)
		result.append(lstm_forward_out)
		result.append(lstm_backward_out)

		return result
