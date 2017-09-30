""" Entry point for different modes. """
import logging
import sys

import os

from model_config import ModelConfig

MODEL_CONFIG = ModelConfig(
    vocab_size=2**13,
)


def main(mode):
    logging.basicConfig(
        format='%(levelname)s:%(asctime)s.%(msecs)03d [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')))

    if mode == 'prep':
        from prep import prep
        prep(MODEL_CONFIG)


if __name__ == '__main__':
    main(sys.argv[1])