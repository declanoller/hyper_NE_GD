from agent_classes import *
from classes import Atom, EvoAgent, HyperEPANN, Population
import argparse



#


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-n', '--name', default='')
    parser.add_argument('-l', '--length', type=int, default=6)
    args = parser.parse_args()

    Population.create_lineage_movie(args.path, fname=args.name, mov_length=args.length)
