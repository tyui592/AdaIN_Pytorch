import argparse

from train import train_network
from evaluate import evaluate_network

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int,
            help='cpu: -1, gpu: 0 ~ n', default=0)

    parser.add_argument('--train', action='store_true',
            help='train flag', default=False)

    parser.add_argument('--content', type=str,
            help='test content image', default=None)

    parser.add_argument('--style', type=str, nargs='+',
            help='test style image', default=None)

    parser.add_argument('--content-dir', type=str,
            help='train content dir', default=None)

    parser.add_argument('--style-dir', type=str,
            help='train style dir', default=None)

    parser.add_argument('--layers', type=int, nargs='+',
            help='layer indices', default=[1, 6, 11, 20])

    parser.add_argument('--style-strength', type=float,
            help='content-style strength interpolation factor, 1: style, 0: content', default=1.0)

    parser.add_argument('--interpolation-weights', type=float, nargs='+',
            help='multi-style interpolation weights', default=None)

    parser.add_argument('--imsize', type=int,
            help='size to resize image', default=512)

    parser.add_argument('--cropsize', type=int,
            help='size to crop image', default=None)

    parser.add_argument('--cencrop', action='store_true',
            help='crop the center region of the image', default=False)

    parser.add_argument('--lr', type=float,
            help='learning rate', default=1e-4)

    parser.add_argument('--max-iter', type=int,
            help='number of iterations to train the network', default=80000)

    parser.add_argument('--batch-size', type=int,
            help='batch size', default=16)

    parser.add_argument('--style-weight', type=float,
            help='style loss weight', default=100)

    parser.add_argument('--check-iter', type=int,
            help='number of iteration to check train logs', default=500)

    parser.add_argument('--load-path', type=str,
            help='model load path', default=None)

    parser.add_argument('--mask', type=str, nargs='+',
            help='mask for multi-style interpolation', default=None)
    
    parser.add_argument('--preserve-color', action='store_true',
            help='flag for color preserved stylization', default=None)
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args= parser.parse_args()

    if args.train:
        train_network(args)

    else:
        stylized_img = evaluate_network(args)
