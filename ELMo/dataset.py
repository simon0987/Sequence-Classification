from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch

class ElmoDataset(Dataset):
	def __init__(self,data,word_vocab,char_vocab):
		self._data = []
		for d in tqdm(data, desc='[*] Indexizing', dynamic_ncols=True):
			for_char = []
			for word in d:
				if word == '<BOS>' or word == '<EOS>':
					for_char.append([char_vocab.vtoi(word)])
				else:
					for_char.append([char_vocab.vtoi(char) for char in word])
			back_char = []
			for word in d[::-1]:
				if word == '<BOS>' or word == '<EOS>':
					back_char.append([char_vocab.vtoi(word)])
				else:
					back_char.append([char_vocab.vtoi(char) for char in word])
			data_dic = {
				'forward_word' : [word_vocab.vtoi(word) for word in d],
				'backward_word' : [word_vocab.vtoi(word) for word in d[::-1]],
				'forward_char' : for_char,
				'backward_char' : back_char
			}
			self._data.append(data_dic)


	def __getitem__(self, index):
		return self._data[index]

	def __len__(self):
		return len(self._data)

def create_collate_fn(word_vocab, char_vocab, max_sent_len, max_word_len):
	word_pad_idx = word_vocab.sp.pad.idx
	char_pad_idx = char_vocab.sp.pad.idx

	# This recursive version can account of arbitrary depth. However, the required stack
	# allocation may harm performance.
	# def pad(batch, max_len, padding):
	#     l, p = max_len[0], padding[0]
	#     for i, b in enumerate(batch):
	#         batch[i] = b[:l]
	#         batch[i] += [[p] for _ in range(l - len(b))]
	#         if len(max_len) > 1:
	#             batch[i] = pad(batch[i], max_len[1:], padding[1:])
	#
	#     return batch

	def pad(batch, max_len, padding, depth=1):
		for i, b in enumerate(batch):
			if depth == 1:
				batch[i] = b[:max_len]
				batch[i] += [padding for _ in range(max_len - len(b))]
			elif depth == 2:
				for j, bb in enumerate(b):
					batch[i][j] = bb[:max_len]
					batch[i][j] += [padding] * (max_len - len(bb))

		return batch

	def collate_fn(batch):
		# batch : [batch_size,sentence_length,word_embedding_length]
		forward_label = [b['forward_word'][1:] for b in batch]
		backward_label = [b['backward_word'][1:] for b in batch]
		forward_char = [b['forward_char'][:-1] for b in batch]
		backward_char = [b['backward_char'][:-1] for b in batch]

		#get the largest dimension of the forward word and get the min of it and the max_sent_len
		max_len = min(max(map(len, forward_label)), max_sent_len)
		forward_label = pad(forward_label, max_len - 1, word_pad_idx)
		backward_label = pad(backward_label, max_len - 1, word_pad_idx)
		forward_char = pad(forward_char, max_len -1 , [char_pad_idx])
		backward_char = pad(backward_char, max_len - 1, [char_pad_idx])
		max_len = min(np.max([[len(w) for w in s] for s in forward_char]), max_word_len)
		forward_char = pad(forward_char, max_len, char_pad_idx, depth=2)
		backward_char = pad(backward_char, max_len, char_pad_idx, depth=2)

		forward_label = torch.tensor(forward_label)
		backward_label = torch.tensor(backward_label)

		label = torch.cat((forward_label,backward_label),0)
		label = label.view(label.size(0)*label.size(1))
		forward_char = torch.tensor(forward_char)
		backward_char = torch.tensor(backward_char)

		return {
			'forward_char' : forward_char,
			'backward_char' : backward_char,
			'label' : label
		}

	return collate_fn


def create_data_loader(dataset, word_vocab, char_vocab, max_sent_len, max_word_len,
					   batch_size, n_workers, shuffle=True):
	collate_fn = create_collate_fn(word_vocab, char_vocab, max_sent_len, max_word_len)
	data_loader = DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
		collate_fn=collate_fn)

	return data_loader