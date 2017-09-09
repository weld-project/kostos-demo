import argparse

from cost_model.python.expressions import *
from cost_model.python.cost_with_bandwidth import *

from cost_model.python.params import CLOCK_FREQUENCY

NSAMPLES = 100

def sparsity_cost(k, n, m):
    cond = EqualTo(Lookup(Vector("A", n*m), Lookup("R", Id("k"))), Literal()) # A is the data, R defines random indices sampled #
    loop = For(k, Id("i"), 1, cond)

    c = cost(loop)
    return c / CLOCK_FREQUENCY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, dest='n')
    parser.add_argument('-m', type=int, required=True, dest='m')
    parser.add_argument('-r', type=int, required=True, dest='r')
    parser.add_argument('-t', type=str, required=True, dest='t')
    args = parser.parse_args()

    if args.t == 's':
        print 'Sparsity:', sparsity_cost(NSAMPLES, args.n, args.m)

if __name__=='__main__':
    main()  
