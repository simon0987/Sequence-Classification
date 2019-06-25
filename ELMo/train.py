import argparse
import random
import sys
from pathlib import Path

import ipdb
import numpy as np
import torch
from box import Box

from .elmo import ELMO
from .dataset import create_data_loader
from common.base_model import BaseModel
from common.base_trainer import BaseTrainer
from common.metrics import Metric
from common.metrics import Accuracy
from common.utils import load_pkl
import torch.nn.functional as F

class Loss(Metric):
    def __init__(self):
        super().__init__()

    def _calculate_loss(self, output, batch):
        raise NotImplementedError

    def reset(self):
        self._sum = 0
        self._n = 0

    def update(self, output, batch):
        loss, loss_sum, n = self._calculate_loss(output, batch)
        self._sum += loss_sum
        self._n += n

        return loss

    @property
    def value(self):
        return self._sum / self._n

class NLLloss(Loss):
    def __init__(self, device, input_key, target_key, weight=None, ignore_index=0,
             reduction='mean'):
        if reduction == 'none':
            raise ValueError('NLLloss: reduction can\'t be none')

        self._device = device
        self._input_key = input_key
        self._target_key = target_key
        self._weight = weight
        self._ignore_index = ignore_index
        self._reduction = reduction
        super().__init__()

    def _set_name(self):
        self.name = 'NLLloss({})'.format(self._target_key)

    def _calculate_loss(self, output, batch):
        _input = output[self._input_key]
        target = batch[self._target_key].to(device=self._device)
        loss = F.nll_loss(
            _input, target, weight=self._weight, ignore_index=self._ignore_index,
            reduction=self._reduction)
        n = (target != self._ignore_index).sum().item()
        loss_sum = loss.item() * (n if self._reduction == 'mean' else 1)

        return loss, loss_sum, n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')
    args = parser.parse_args()

    return vars(args)

class Model(BaseModel):
    def _create_net_and_optim(self, word_vocab, char_vocab, net_cfg, optim_cfg):
        net = ELMO(word_vocab, char_vocab)
        net.to(device=self._device)

        optim = getattr(torch.optim, optim_cfg.algo)
        optim = optim(
            filter(lambda p: p.requires_grad, net.parameters()), **optim_cfg.kwargs)

        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, max_sent_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_sent_len = max_sent_len

    def _run_batch(self, batch):
        forward_char = batch['forward_char'].to(device=self._device)
        backward_char = batch['backward_char'].to(device=self._device)
        # if self._elmo_embedder and self._elmo_embedder.ctx_emb_dim > 0:
        #     text_ctx_emb = self._elmo_embedder(batch['text_orig'], self._max_sent_len)
        #     text_ctx_emb = torch.tensor(text_ctx_emb, device=self._device)
        # else:
        #     text_ctx_emb = torch.empty(
        #         (*text_word.shape, 0), dtype=torch.float32, device=self._device)
        # text_pad_mask = batch['text_pad_mask'].to(device=self._device)
        logits = self._model(forward_char, backward_char)
        label = logits.max(dim=1)[1]

        return {
            'logits': logits,
            'label': label
        }


def main(model_dir):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)
    print(
        '[-] Model checkpoints and training log will be saved to {}\n'
        .format(model_dir))

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = model_dir / 'log.csv'
    ckpt_dir = model_dir / 'ckpts'
    if any([p.exists() for p in [log_path, ckpt_dir]]):
        print('[!] Directory already contains saved ckpts/log')
        exit(1)
    ckpt_dir.mkdir()

    print('[*] Loading vocabs and datasets from {}'.format(cfg.dataset_dir))
    dataset_dir = Path(cfg.dataset_dir)
    word_vocab = load_pkl(dataset_dir / 'word.pkl')
    char_vocab = load_pkl(dataset_dir / 'char.pkl')
    train_dataset = load_pkl(dataset_dir / 'train.pkl')
    dev_dataset = load_pkl(dataset_dir / 'dev.pkl')

    print('[*] Creating train/dev data loaders\n')
    if cfg.data_loader.batch_size % cfg.train.n_gradient_accumulation_steps != 0:
        print(
            '[!] n_gradient_accumulation_steps({}) is not a divider of '
            .format(cfg.train.n_gradient_accumulation_steps),
            'batch_size({})'.format(cfg.data_loader.batch_size))
        exit(1)
    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    train_data_loader = create_data_loader(
        train_dataset, word_vocab, char_vocab, **cfg.data_loader)
    dev_data_loader = create_data_loader(
        dev_dataset, word_vocab, char_vocab, **cfg.data_loader)


    print('[*] Creating model\n')
    # cfg.net.n_ctx_embs = cfg.elmo_embedder.n_ctx_embs if cfg.use_elmo else 0
    # cfg.net.ctx_emb_dim = cfg.elmo_embedder.ctx_emb_dim if cfg.use_elmo else 0
    model = Model(device, word_vocab, char_vocab,cfg.net,cfg.optim)

    trainer = Trainer(
        cfg.data_loader.max_sent_len, device, cfg.train,
        train_data_loader, dev_data_loader, model,
        [NLLloss(device, 'logits', 'label',ignore_index = word_vocab.sp.pad.idx)], [Accuracy(device, 'label')],
        log_path, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
