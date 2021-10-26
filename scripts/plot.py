import argparse
import sys
sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments

parser = argparse.ArgumentParser()
parser.add_argument("folder", nargs="+", type=str, default="./.checkpoints/CMA ppo 2/3")
parser.add_argument("-y", action="store_true", default=False)
parser.add_argument("-x", action="store_true", default=False)
parser.add_argument("-ylim", nargs=2, type=float, default=None)
parser.add_argument("--names", nargs="+", type=str, default=None)
parser.add_argument("--rawmode", action="store_false", default=True)
args = parser.parse_args()

if len(args.folder) == 1:
	plot_experiment(args.folder[0], logyscale=args.y, logxscale=args.x, ylim=args.ylim)
else:
	compare_experiments(args.folder, args.names, logyscale=args.y, logxscale=args.x, ylim=args.ylim, cleanPlotMode=args.rawmode)