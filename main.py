""" Entry point for different modes. """

import sys


def main(mode):
    if mode == 'prep':
        from prep import prep
        prep()


if __name__ == '__main__':
    main(sys.argv[1])