import logging
import random

import numpy

from model import GruModel, build_model, get_experiment
from model_config import ModelConfig
from bpe import Encoder


def train(cfg):
    # type: (ModelConfig) -> None
    """ Trains a model! """
    logging.info("Loading encoder...")
    encoder = Encoder.load('current_encoder.json')

    logging.info("Building model...")
    model = build_model(cfg, encoder.word_vocab[cfg.start_token])

    logging.info("Logging model config...")
    get_experiment().log_multiple_params(cfg.__dict__)

    logging.info("Loading x...")
    x = numpy.loadtxt('./data/x.csv')
    logging.info("Loading y...")
    y = numpy.loadtxt('./data/y_title.csv')

    shuffle_idx = list(range(len(x)))

    logging.info("Training model...")
    for epoch in range(10000):
        print_examples(cfg, encoder, model, x)
        random.shuffle(shuffle_idx)
        logging.info("Training in epoch {}".format(epoch))
        model.train_epoch(x[shuffle_idx], y[shuffle_idx],
                          callback=lambda loss: get_experiment().log_metric('loss', loss))
        get_experiment().log_epoch_end(epoch)


def print_examples(cfg, encoder, model, x):
    # type: (ModelConfig, Encoder, GruModel, numpy.ndarray) -> None
    """ Pick random inputs, decodes and prints them, and decodes and prints their predictions. """
    try:
        picks = random.sample(list(range(len(x))), cfg.batch_size)
        x_sample = x[picks]
        decoded_x = list(encoder.inverse_transform(_x[::-1] for _x in x_sample))

        y = model.predict(x_sample)
        decoded_y = list(encoder.inverse_transform(y))

        step = 1
        for _x, _y in zip(decoded_x, decoded_y):
            logging.info("Input {}: {}".format(step, _x))
            logging.info("Response {}: {}".format(step, _y))
            step += 1

            if step > 4:
                break
    except ValueError:
        pass
