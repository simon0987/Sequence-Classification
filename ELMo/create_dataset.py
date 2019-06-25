import argparse
from collections import Counter
import pickle
import sys
from pathlib import Path
import ipdb
from tqdm import tqdm
from .dataset import ElmoDataset
from common.vocab import Vocab
from box import Box
import random

def load_choose_data(filepath,sentence_num):
	print("[*] start loading data.")
	with open(filepath,"r",encoding="UTF-8") as f:
		all_line = f.readlines()
	#random choose sentence from 1-len(all_line)
	randomlist = random.sample(range(0,len(all_line)), sentence_num)
	sentences = []
	for random_num in tqdm(randomlist,desc='[*] choosing 410000 sentences', dynamic_ncols=True):
		sentences.append(all_line[random_num])

	return sentences

def create_vocab(data,cfg,dataset_dir):
	print('[*] Creating word vocab')
	dict_words = Counter()
	for m, d in data.items():
		for words in tqdm(d, desc='[*] creating word vocab', dynamic_ncols=True):
			dict_words.update(words)
	dict_words = Counter({word : dict_words[word] for word in dict_words if dict_words[word] > 3})
	tokens = [w for w, _ in dict_words.most_common(cfg.word.size)]
	word_vocab = Vocab(tokens,**cfg.word)
	print("[*] The word vocabulary size is " + str(word_vocab.__len__()))
	word_vocab_path = (dataset_dir / 'word.pkl')
	with word_vocab_path.open(mode='wb') as f:
		pickle.dump(word_vocab,f)
	print('[-] Word vocab saved at {}\n'.format(word_vocab_path))

	print('[*] Creating char vocab')
	dict_chars = Counter()
	for m,d in data.items():
		for words in tqdm(d, desc='[*] creating char vocab', dynamic_ncols=True):
			for word in words:
				if word == '<BOS>' or word == '<EOS>':
					dict_chars.update([word])
					continue
				dict_chars.update(word)
	dict_chars = Counter({char : dict_chars[char] for char in dict_chars if dict_chars[char] > 1000})
	tokens = [c for c, _ in dict_chars.most_common(cfg.char.size)]
	char_vocab = Vocab(tokens,**cfg.char)
	print("[*] The char vocabulary size is " + str(char_vocab.__len__()))
	char_vocab_path = (dataset_dir / 'char.pkl')
	with char_vocab_path.open(mode='wb') as f:
		pickle.dump(char_vocab, f)
	print('[-] Char vocab saved to {}\n'.format(char_vocab_path))

	return word_vocab, char_vocab


def create_sample(all_line,sentence_num):
	all_line = ["<BOS> " + line + " <EOS>" for line in all_line]
	data = {}
	train = []
	dev = []
	for sentence in tqdm(all_line[:sentence_num-10000], desc='[*] creating train samples', dynamic_ncols=True):
		words = sentence.split()
		for i in range(0,len(words),64):
			train.append(words[i:64+i])
	for sentence in tqdm(all_line[sentence_num-10000:sentence_num], desc='[*] creating dev samples', dynamic_ncols=True):
		words = sentence.split()
		for i in range(0,len(words),64):
			dev.append(words[i:64+i])
	data['train'] = train
	data['dev'] = dev
	return data

def create_dataset(data,word_vocab,char_vocab,dataset_dir):
	for m,d in data.items():
		print('[*] Creating {} dataset'.format(m))
		dataset = ElmoDataset(d, word_vocab, char_vocab)
		dataset_path = (dataset_dir / '{}.pkl'.format(m))
		with dataset_path.open(mode='wb') as f:
			pickle.dump(dataset, f)
		print('[-] dataset saved to {}\n'.format(m.capitalize(),dataset_path))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_dir', type=Path, help='Target dataset directory')
	args = parser.parse_args()

	return vars(args)

def main(dataset_dir):
	try:
		cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
	except FileNotFoundError:
		print('[!] Dataset directory({}) must contain config.yaml'.format(dataset_dir))
		exit(1)
	print('[-] Vocabs and datasets will be saved to {}\n'.format(dataset_dir))
	output_files = ['word.pkl', 'char.pkl']
	if any([(dataset_dir / p).exists() for p in output_files]):
		print('[!] Directory already contains saved vocab/dataset')
		exit(1)
	data_dir = Path(cfg.data_dir)
	all_line = load_choose_data(cfg.data_dir + "/corpus_tokenized.txt",410000)
	data = create_sample(all_line,410000) 
	word_vocab, char_vocab = create_vocab(data, cfg.vocab, dataset_dir)
	create_dataset(data, word_vocab, char_vocab, dataset_dir)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
