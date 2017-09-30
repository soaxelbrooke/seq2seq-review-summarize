""" Loading and partitioning of data before model training and eval. """
import logging
import os
import json
import random

import numpy
from bpe import Encoder
from numpy import ndarray
from tqdm import tqdm

from model_config import ModelConfig

try:
    from typing import Iterator, Union, Dict, List, Tuple
except ImportError:
    # Pre-Python3.5, alas
    pass

TOTAL_REVIEWS = 582133


def review_file_iter():
    # type: () -> Iterator[str]
    """ Iterates over all review file paths """
    category_dirs = os.listdir('data')
    for category_dir in category_dirs:
        if '.' not in category_dir:
            for fname in os.listdir('data' + os.sep + category_dir):
                yield 'data' + os.sep + category_dir + os.sep + fname


def review_iter():
    """ Loads and partitions data. """
    for fpath in review_file_iter():
        with open(fpath) as infile:
            all_info = json.load(infile)
        product_info = all_info['ProductInfo']
        reviews = all_info['Reviews']

        if not (product_info.get('Price') or 'Unavailable').startswith('$'):
            logging.warning("Skipping product {} due to price '{}'".format(
                product_info['ProductID'], product_info['Price']))
            continue

        for review in reviews:
            if not isinstance(review['Content'], str):
                continue
            try:
                review['Price'] = float(product_info['Price'][1:].replace(',', ''))
                review['Overall'] = float(review['Overall'])
                review['Features'] = product_info['Features']
                review['ProductName'] = product_info['Name']
                review['Category'] = fpath.split(os.sep)[0]
                yield review
            except (ValueError, AttributeError) as e:
                logging.warning('Skipping review {} due to {}'.format(review['ReviewID'], e))


def reviews_to_x_y(model_cfg, enc, reviews):
    # type: (ModelConfig, Encoder, List[Dict[str, Union[int, str, float]]]) -> Tuple[ndarray, ndarray, ndarray]
    """ Turns a review into the X and Y vecs, where product review content prefixed by product
        category, Y is a tuple of the review title, and the the overall score of the review.
    """
    x = numpy.array(list(enc.transform(['__' + r['Category'] + ' ' + r['Content'] for r in reviews],
                                       fixed_length=model_cfg.review_len)),
                    dtype='int32')
    y_title = numpy.array(list(enc.transform([r['Title'] for r in reviews],
                                             fixed_length=model_cfg.summary_len)),
                          dtype='int32')
    y_overall = numpy.array([[r['Overall']] for r in tqdm(reviews)], dtype='int32')

    return x, y_title, y_overall


def prep(cfg: ModelConfig):
    """ Prepares data, fits new encoder """
    logging.info('Loading reviews...')
    reviews = list(tqdm(review_iter(), total=TOTAL_REVIEWS))

    print(len(reviews))
    logging.info("Shuffling reviews...")
    random.shuffle(reviews)

    logging.info("Fitting BPE encoder with vocab size {}...".format(cfg.vocab_size))
    encoder = Encoder(vocab_size=cfg.vocab_size, silent=False, required_tokens=[cfg.start_token])
    encoder.fit([review['Content'] for review in tqdm(reviews)])
    encoder.save('current_encoder.json')

    logging.info("Encoding reviews and building x, y vectors...")
    x, y_title, y_overall = reviews_to_x_y(cfg, encoder, reviews)

    logging.info("Saving X to ./data/x.csv...")
    numpy.savetxt('data/x.csv', x)
    logging.info("Saving Y titles to ./data/y_title.csv...")
    numpy.savetxt('data/y_title.csv', y_title)
    logging.info("Saving Y overall to ./data/y_overall.csv...")
    numpy.savetxt('data/y_overall.csv', y_overall)

    logging.info("Done with data prep!")
