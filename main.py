""" Entry point for different modes. """
import logging
import sys

import os

from model_config import ModelConfig

MODEL_CONFIG = ModelConfig(
    vocab_size=2**13,
    use_cuda=True,
    embed_size=100,
    review_len=100,
    summary_len=10,
    batch_size=int(os.environ.get('BATCH_SIZE', '16')),
    context_size=100,
    start_token='<start/>',
    attention_method='dot',
)


def main(mode):
    logging.basicConfig(
        format='%(levelname)s:%(asctime)s.%(msecs)03d [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')))

    if mode == 'prep':
        from prep import prep
        prep(MODEL_CONFIG)

    if mode == 'train':
        from train import train
        train(MODEL_CONFIG)


if __name__ == '__main__':
    main(sys.argv[1])