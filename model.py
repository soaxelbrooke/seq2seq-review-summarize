import logging
from collections import deque
from typing import Optional

import toolz
import torch
from comet_ml import Experiment
from numpy import ndarray
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_config import ModelConfig
import random
from torch import optim
from tqdm import tqdm
import numpy as np
import os


@toolz.memoize
def get_experiment():
    # type: () -> Optional[Experiment]
    """ Memoized constructor for comet.ml instrumentation. """
    comet_ml_key = os.environ.get('COMET_ML_KEY')
    if comet_ml_key is not None:
        return Experiment(api_key=comet_ml_key, log_code=True)
    else:
        logging.info('No comet.ml API key (COMET_ML_KEY), skipping recording...')
        return None


def build_model(cfg, start_idx):
    # type: (ModelConfig, int) -> GruModel
    """ Builds a bomb ass model """
    shared_embedding = build_shared_embedding(cfg)
    encoder = GruEncoder(cfg, shared_embedding, 2)
    decoder = AttnDecoder(cfg, shared_embedding)

    if cfg.use_cuda:
        encoder.cuda()
        decoder.cuda()

    logging.info("Built model:")
    logging.info('Encoder:\n' + str(encoder))
    logging.info('Decoder:\n' + str(decoder))

    return GruModel(cfg, encoder, decoder, shared_embedding, start_idx)


def build_shared_embedding(cfg):
    """ Builds embedding to be used by encoder and decoder """
    # type: ModelConfig -> nn.Embedding
    return nn.Embedding(cfg.vocab_size, cfg.embed_size)


class GruModel:
    def __init__(self, model_cfg, encoder, decoder, embedding, start_idx):
        # type: (ModelConfig, GruEncoder, AttnDecoder, nn.Embedding, int) -> None
        self.cfg = model_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.start_idx = start_idx

        self.gradient_clip = 5.0
        self.teacher_force_ratio = 0.5
        self.learning_rate = self.cfg.learning_rate

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.NLLLoss()

    def teacher_should_force(self):
        return random.random() < self.teacher_force_ratio

    def train_epoch(self, train_x, train_y, callback=lambda loss: loss):
        # type: (ndarray, ndarray) -> float
        """ Trains a single epoch. Returns training loss. """
        progress = tqdm(total=len(train_x))
        loss_queue = deque(maxlen=100)
        train_x = train_x.astype('int64')
        train_y = train_y.astype('int64')
        idx_iter = zip(range(0, len(train_x) - self.cfg.batch_size, self.cfg.batch_size),
                       range(self.cfg.batch_size, len(train_x), self.cfg.batch_size))

        for start, end in idx_iter:
            x_batch = torch.LongTensor(train_x[start:end])
            y_batch = torch.LongTensor(train_y[start:end])

            if self.cfg.use_cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            loss = self._train_inner(
                Variable(x_batch.view(-1, self.cfg.batch_size)),
                Variable(y_batch.view(-1, self.cfg.batch_size)),
            )
            loss_queue.append(loss)
            mean_loss = np.mean(loss_queue)
            progress.set_postfix(loss=mean_loss, refresh=False)
            callback(mean_loss)
            progress.update(self.cfg.batch_size)

        return np.mean(loss_queue)

    def _train_inner(self, input_var_batch, target_var_batch):
        # type: (ndarray, ndarray) -> float
        C = self.cfg.context_size
        B = self.cfg.batch_size

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0

        enc_hidden_state = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_var_batch, enc_hidden_state)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # reshape to (B, R, C)
        context = encoder_hidden[1, :, :]
        decoder_hidden = context

        decoder_input = Variable(torch.LongTensor([[self.start_idx]] * self.cfg.batch_size))

        if self.cfg.use_cuda:
            decoder_input = decoder_input.cuda()

        should_use_teacher = self.teacher_should_force()
        for input_idx in range(self.cfg.summary_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.view(B, 1), context.view(B, C), decoder_hidden.view(B, C), encoder_outputs)
            loss += self.loss_fn(decoder_output.squeeze(1), target_var_batch[input_idx, :])

            if should_use_teacher:
                decoder_input = target_var_batch[input_idx, :]

            else:
                # Get the highest values and their indexes over axis 1
                top_vals, top_idxs = decoder_output.data.topk(1)
                decoder_input = Variable(top_idxs.squeeze())

        loss.backward()
        # nn.utils.clip_grad_norm(self.encoder.parameters(), self.gradient_clip)
        # nn.utils.clip_grad_norm(self.decoder.parameters(), self.gradient_clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data.sum() / self.cfg.summary_len

    def predict(self, x_minibatch):
        # type: (ndarray) -> ndarray
        """ Predicts the response for a mini-batch """
        C = self.cfg.context_size
        B = self.cfg.batch_size

        assert len(x_minibatch.shape) == 2, 'minibatch should be two dimensional'
        assert x_minibatch.shape[1] == self.cfg.review_len

        x_tensor = torch.from_numpy(x_minibatch.astype('int64'))

        if self.cfg.use_cuda:
            x_tensor = x_tensor.cuda()

        x = Variable(x_tensor.view(-1, self.cfg.batch_size))

        y = np.array([[self.start_idx]] * self.cfg.batch_size)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        enc_hidden_state = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(x, enc_hidden_state)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # reshape to (B, R, C)
        context = encoder_hidden[1, :, :]
        decoder_hidden = context

        decoder_input = Variable(torch.LongTensor([[self.start_idx]] * self.cfg.batch_size))

        if self.cfg.use_cuda:
            decoder_input = decoder_input.cuda()

        for input_idx in range(self.cfg.summary_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.view(B, 1), context.view(B, C), decoder_hidden.view(B, C), encoder_outputs)

            # Get the highest values and their indexes over axis 1
            top_vals, top_idxs = decoder_output.data.topk(1)
            decoder_input = Variable(top_idxs.squeeze())
            y = np.hstack((y, decoder_input.data.cpu().view(-1, 1).numpy()))

        return y


class GruEncoder(nn.Module):
    def __init__(self, seq2seq_params, embedding, n_layers=1):
        # type: (ModelConfig, nn.Embedding, int) -> None
        super(GruEncoder, self).__init__()
        self.cfg = seq2seq_params
        self.n_layers = n_layers

        self.embedding = embedding
        self.rnn = nn.GRU(
            input_size=self.cfg.embed_size,
            hidden_size=self.cfg.context_size,
            num_layers=self.n_layers,
        )

    def forward(self, word_idxs, hidden_state):
        embedded = self.embedding(word_idxs) \
            .view(self.cfg.review_len, self.cfg.batch_size, self.cfg.embed_size)

        return self.rnn(embedded, hidden_state)

    def init_hidden(self):
        hidden = Variable(torch.randn(self.n_layers, self.cfg.batch_size, self.cfg.context_size))
        return hidden.cuda() if self.cfg.use_cuda else hidden


class Attn(nn.Module):
    def __init__(self, cfg):
        # type: (ModelConfig) -> None
        super(Attn, self).__init__()

        self.cfg = cfg
        C = self.cfg.context_size
        B = self.cfg.batch_size
        R = self.cfg.review_len

        self.attn_linear = nn.Linear(R * C, C)
        self.hidden_size = cfg.context_size

    def forward(self, decoder_hidden, encoder_outputs):
        # type: (Variable, Variable) -> Variable
        # decoder_hidden: (B, C)
        # encoder_outputs: (B, R, C)
        # return: (B, C)
        C = self.cfg.context_size
        B = self.cfg.batch_size
        R = self.cfg.review_len

        softmax = F.softmax(torch.bmm(decoder_hidden.unsqueeze(1),
                                      encoder_outputs.permute(0, 2, 1)).sum(1))

        softmax_scaled = torch.mul(softmax.unsqueeze(2).repeat(1, 1, C), encoder_outputs) \
            .view(B, R, C)
        # return softmax_scaled.sum(1)
        return self.attn_linear(softmax_scaled.view(B, R * C)).view(B, C)



class AttnDecoder(nn.Module):
    def __init__(self, cfg, shared_embedding, n_layers=1, dropout_p=0.1):
        # type: (ModelConfig, Embedding, int, float) -> None
        super(AttnDecoder, self).__init__()

        # Keep parameters for reference
        self.cfg = cfg
        B = self.cfg.batch_size
        C = self.cfg.context_size
        E = self.cfg.embed_size
        V = self.cfg.vocab_size

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = shared_embedding
        self.gru = nn.GRU(E + C + C, C, n_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(C, V)

        # Choose attention model
        if cfg.attention_method != 'none':
            self.attn = Attn(cfg)

    def forward(self, word_input, context, last_decoder_hidden, encoder_outputs):
        # type: (Variable, Variable, Variable, Variable) -> Variable
        """
        word_input: (B)
        context: (B, C)
        last_decoder_hidden: (B, 1, C)
        encoder_outputs: (B, R, C)

        return shape: [(B, 1, V), (B, 1, C)
        """
        B = self.cfg.batch_size
        C = self.cfg.context_size
        E = self.cfg.embed_size
        V = self.cfg.vocab_size

        word_embed = self.embedding(word_input).view(B, 1, E)
        attention = self.attn(last_decoder_hidden, encoder_outputs).view(B, 1, C)
        decoder_input = torch.cat([word_embed, context.unsqueeze(1), attention], 2).view(B, 1, E + C + C)

        decoder_output, decoder_hidden = self.gru(decoder_input, last_decoder_hidden.unsqueeze(0))
        output = F.log_softmax(self.out(decoder_output.view(B, 1, C)).view(B, 1, V))

        return output, decoder_hidden
