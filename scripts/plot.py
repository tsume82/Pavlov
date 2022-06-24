import argparse
import sys

sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments

parser = argparse.ArgumentParser()
parser.add_argument("dir_or_file", nargs="+", type=str)
parser.add_argument("-y", action="store_true", default=False)
parser.add_argument("-x", action="store_true", default=False)
parser.add_argument("-ylim", nargs=2, type=float, default=None)
parser.add_argument("--names", nargs="+", type=str, default=["1", "2"])
parser.add_argument("--title", type=str, default="Compare Experiments")
parser.add_argument("--save", "-s", type=str, default="")
parser.add_argument("--plotmode", type=str, default="std")
parser.add_argument("--no_actions", action="store_false", default=True)
args = parser.parse_args()

if len(args.dir_or_file) == 1:
    plot_experiment(args.dir_or_file[0], logyscale=args.y, logxscale=args.x, ylim=args.ylim)
else:
    compare_experiments(
        args.dir_or_file,
        args.names,
        title=args.title,
        logyscale=args.y,
        logxscale=args.x,
        ylim=args.ylim,
        plotMode=args.plotmode,
        save=args.save,
        show_actions=args.no_actions,
    )
