import logging

import numpy

from model import GruModel, build_model
from model_config import ModelConfig
from bpe import Encoder


def train(cfg):
    # type: (ModelConfig) -> None
    """ Trains a model! """
    logging.info("Loading encoder...")
    encoder = Encoder.load('current_encoder.json')

    logging.info("Building model...")
    model = build_model(cfg, encoder.word_vocab[cfg.start_token])

    logging.info("Loading x...")
    x = numpy.loadtxt('./data/x.csv')
    logging.info("Loading y...")
    y = numpy.loadtxt('./data/y_title.csv')

    logging.info("Training model...")
    for epoch in range(10000):
        logging.info("Training in epoch {}".format(epoch))
        model.train_epoch(x, y)
