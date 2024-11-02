#config file for training phase and testing phase

import argparse
choices = [1, 2, 3]

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_classificator', type=str, default=0.001)
parser.add_argument('--min_loss', type=float, default=1e10)
parser.add_argument('--result', type=str, default='results')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stopping', type=int, default=15)
parser.add_argument('--drive', action="store_true", default=False)
parser.add_argument('--checkpoints', action='store_true', default=False)
args = parser.parse_args()
