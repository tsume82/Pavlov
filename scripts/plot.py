import argparse
import sys
sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str, default="./.checkpoints/CMA ppo 2/3")
parser.add_argument("-y", action="store_true", default=False)
parser.add_argument("-x", action="store_true", default=False)
parser.add_argument("-ylim", nargs=2, type=float, default=None)
args = parser.parse_args()

plot_experiment(args.folder, logyscale=args.y, logxscale=args.x, ylim=args.ylim)