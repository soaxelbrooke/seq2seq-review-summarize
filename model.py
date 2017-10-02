from collections import deque
from datetime import datetime

import torch
from numpy import ndarray
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_config import ModelConfig
import random
from torch import optim
from tqdm import tqdm
import numpy as np

TIME = False
last_time = datetime.now()


def time(message):
    if not TIME:
        return
    global last_time
    torch.cuda.synchronize()
    now = datetime.now()
    print('[{}] - Time: {}'.format(message, now - last_time))
    last_time = now


def build_model(cfg, start_idx):
    # type: (ModelConfig, int) -> GruModel
    """ Builds a bomb ass model """
    shared_embedding = build_shared_embedding(cfg)
    encoder = GruEncoder(cfg, shared_embedding, 2)
    decoder = AttnDecoder(cfg, shared_embedding)

    if cfg.use_cuda:
        encoder.cuda()
        decoder.cuda()

    print(encoder)
    print(decoder)

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
        self.learning_rate = 0.001

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.NLLLoss()

    def teacher_should_force(self):
        return random.random() < self.teacher_force_ratio

    def train_epoch(self, train_x, train_y):
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
            progress.set_postfix(loss=np.mean(loss_queue), refresh=False)
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
        decoder_hidden = encoder_hidden[1, :, :]

        decoder_context = Variable(torch.zeros(1, self.cfg.batch_size, self.cfg.context_size))
        decoder_input = Variable(torch.LongTensor([[self.start_idx]] * self.cfg.batch_size))

        if self.cfg.use_cuda:
            decoder_context = decoder_context.cuda()
            decoder_input = decoder_input.cuda()

        should_use_teacher = self.teacher_should_force()
        for input_idx in range(self.cfg.summary_len):
            # print(decoder_input.size(), decoder_hidden.size())
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.view(B, 1), decoder_hidden.view(B, C), encoder_outputs)
            loss += self.loss_fn(decoder_output.squeeze(1), target_var_batch[input_idx, :])

            if should_use_teacher:
                decoder_input = target_var_batch[input_idx, :]
            # torch.Size([16, 1]) torch.Size([16, 100]) torch.Size([100, 16, 100])

            else:
                # Get the highest values and their indexes over axis 1
                top_vals, top_idxs = decoder_output.data.topk(1)
                decoder_input = Variable(top_idxs.squeeze())

        time('pre-loss-backward')
        loss.backward()
        time('post-loss-backward')
        nn.utils.clip_grad_norm(self.encoder.parameters(), self.gradient_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(), self.gradient_clip)
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
        decoder_hidden = encoder_hidden[1, :, :]

        decoder_context = Variable(torch.zeros(1, self.cfg.batch_size, self.cfg.context_size))
        decoder_input = Variable(torch.LongTensor([[self.start_idx]] * self.cfg.batch_size))

        if self.cfg.use_cuda:
            decoder_context = decoder_context.cuda()
            decoder_input = decoder_input.cuda()

        for input_idx in range(self.cfg.summary_len):
            # print(decoder_input.size(), decoder_hidden.size())
            decoder_output, decoder_hidden = self.decoder(
                decoder_input.view(B, 1), decoder_hidden.view(B, C), encoder_outputs)

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


class GruDecoder(nn.Module):
    def __init__(self, seq2seq_params, embedding, n_layers, dropout_p=0.1):
        # type: (ModelConfig, nn.Embedding, int, float) -> None
        super(GruDecoder, self).__init__()

        self.cfg = seq2seq_params
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = embedding
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.GRU(
            input_size=self.cfg.embed_size,
            hidden_size=self.cfg.context_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p,
        )
        self.out = nn.Linear(self.cfg.context_size, self.cfg.vocab_size)

    def forward(self, word_idx_slice, last_hidden_state):
        """ Processes a single slice of the minibatch - a single word per row """
        embedded_words = self.embedding(word_idx_slice) \
            .view(1, self.cfg.batch_size, self.cfg.embed_size)
        post_dropout_words = self.dropout(embedded_words)

        output, hidden_state = self.rnn(post_dropout_words, last_hidden_state)
        word_dist = F.log_softmax(self.out(output.squeeze(0)))

        return word_dist, hidden_state


class Attn(nn.Module):
    def __init__(self, cfg):
        # type: (ModelConfig) -> None
        super(Attn, self).__init__()

        self.cfg = cfg
        self.method = cfg.attention_method
        self.hidden_size = cfg.context_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, cfg.context_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, cfg.context_size)
            self.other = nn.Parameter(torch.FloatTensor(1, cfg.context_size))

    def forward(self, decoder_hidden, encoder_outputs):
        # type: (Variable, Variable) -> Variable
        # decoder_hidden: (1, batch_size, context_size)
        # encoder_outputs: (review_len, batch_size, context_size)
        # return: (1, batch_size, context_size)

        # Dimensions of attn_energies are (batch, decoder_output, context_idx)
        time('forward-start')
        attn_energies = Variable(torch.zeros(self.cfg.batch_size, self.cfg.review_len))
        if self.cfg.use_cuda:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for output_idx in range(self.cfg.review_len):
            batch_softmaxes = []

            for batch_idx in range(self.cfg.batch_size):
                batch_softmax = self.score(decoder_hidden[batch_idx],
                                           encoder_outputs[:, batch_idx, output_idx])
                batch_softmaxes.append(torch.mul(encoder_outputs[:, batch_idx, output_idx],
                                                 batch_softmax))

            attn_energies += F.softmax(torch.cat(batch_softmaxes).view(self.cfg.batch_size, -1))

        time('attention')
        return attn_energies

    def score(self, decoder_hidden, encoder_output_slice):
        # Scores for a single encoder_output
        # print(decoder_hidden.size(), encoder_output_slice.size())
        if self.method == 'dot':
            return decoder_hidden.dot(encoder_output_slice)

        elif self.method == 'general':
            return decoder_hidden.dot(self.attn(encoder_output_slice))

        elif self.method == 'concat':
            return self.other.dot(self.attn(torch.cat((decoder_hidden, encoder_output_slice), 1)))


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
        self.gru = nn.GRU(C + E, C, n_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(C, V)

        # Choose attention model
        if cfg.attention_method != 'none':
            self.attn = Attn(cfg)

    def forward(self, word_input, last_decoder_hidden, encoder_outputs):
        # type: (Variable, Variable, Variable, Variable) -> Variable
        """
        word_input: (B)
        last_decoder_hidden: (B, 1, C)
        encoder_outputs: (B, R, C)

        return shape: [(B, 1, V), (B, 1, C)
        """
        # print(word_input.size(), last_decoder_hidden.size(), encoder_outputs.size())
        B = self.cfg.batch_size
        C = self.cfg.context_size
        E = self.cfg.embed_size
        V = self.cfg.vocab_size

        word_embed = self.embedding(word_input).view(B, 1, E)
        attention = self.attn(last_decoder_hidden, encoder_outputs).view(B, 1, C)
        decoder_input = torch.cat([word_embed, attention], 2).view(B, 1, E + C)

        decoder_output, decoder_hidden = self.gru(decoder_input, last_decoder_hidden.unsqueeze(0))
        output = F.log_softmax(self.out(decoder_output.view(B, 1, C)).view(B, 1, V))

        return output, decoder_hidden


class AttnDecoderOld(nn.Module):
    def __init__(self, cfg, shared_embedding, n_layers=1, dropout_p=0.1):
        # type: (ModelConfig, Embedding, int, float) -> None
        super(AttnDecoderOld, self).__init__()

        # Keep parameters for reference
        self.cfg = cfg
        self.attn_model = cfg.attention_method
        self.hidden_size = cfg.context_size
        self.output_size = cfg.vocab_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = shared_embedding
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

        # Choose attention model
        if cfg.attention_method != 'none':
            self.attn = Attn(cfg)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # word_input: (batch_size, 1)
        # last_context: (1, batch_size, context_size)
        # last_hidden: (1, batch_size, context_size)
        # encoder_outputs: (review_len, batch_size, context_size)

        # word_embedded: (1, batch_size, embed_size)
        word_embedded = self.embedding(word_input).view(1, self.cfg.batch_size, -1)

        # (1, batch_size, embed_size), (1, batch_size, context_size) ->
        #     (1, batch_size, embed_size + context_size)
        rnn_input = torch.cat([word_embedded, last_context], 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder
        # outputs
        # attn_weights: (1, 128, 100)
        attn_weights = self.attn(rnn_output, encoder_outputs) \
            .view(1, self.cfg.batch_size, self.cfg.context_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
