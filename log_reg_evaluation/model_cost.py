import argparse

from cost_model.python.expressions import *
from cost_model.python.cost_with_bandwidth import *

from cost_model.python.params import CLOCK_FREQUENCY

# Used to generate unique loop index variable names.
tempvar = [100]

def transpose(T, A, n, m):
    
    var_suffix = str(tempvar[0])
    tempvar[0] += 1

    i = "i" + var_suffix
    j = "j" + var_suffix

    update = Add(Lookup(T, [Id(i), Id(j)]), Lookup(A, [Id(j), Id(i)]))
    return For(n, Id(i), 1, For(m, Id(j), 1, update))

# Models a Blocked (n x m) (m x l) matrix multiply.
def blocked_matrix_multiply(A, B, C, n, m, l, blocksize=32):

    var_suffix = str(tempvar[0])
    tempvar[0] += 1

    i = "i" + var_suffix
    j = "j" + var_suffix
    k = "k" + var_suffix
    ii = "ii" + var_suffix
    jj = "jj" + var_suffix
    kk = "kk" + var_suffix

    kk_loop = For(blocksize, Id(kk), 1, Add(Lookup(C, [Id(ii), Id(jj)]),
                                    Multiply(Lookup(A, [Id(ii), Id(kk)]),
                                             Lookup(B, [Id(kk), Id(jj)]))))

    jj_loop = For(blocksize, Id(jj), 1, kk_loop)
    ii_loop = For(blocksize, Id(ii), 1, jj_loop)
    k_loop = For(m, Id(k), blocksize, ii_loop)
    j_loop = For(l, Id(j), blocksize, k_loop)
    i_loop = For(n, Id(i), blocksize, j_loop)

    return i_loop

# Models a naive (unblocked) (n x m) (m x l) matrix multiply.
def naive_matrix_multiply(A, B, C, n, m, l):

    var_suffix = str(tempvar[0])
    tempvar[0] += 1

    i = "i" + var_suffix
    j = "j" + var_suffix
    k = "k" + var_suffix

    k_loop = For(m, Id(k), 1, Add(Lookup(C, [Id(i), Id(j)]),
                                    Multiply(Lookup(A, [Id(i), Id(k)]),
                                             Lookup(B, [Id(k), Id(j)]))))
    j_loop = For(l, Id(j), 1, k_loop)
    i_loop = For(n, Id(i), 1, j_loop)
    return i_loop

def matrix_vector_dot(matrix, vector, length, row_idx, transposed=False):

    var_suffix = str(tempvar[0])
    tempvar[0] += 1

    j = "j" + var_suffix

    matrix_indices = [Id(row_idx), Id(j)]
    if transposed:
        matrix_indices.reverse()

    expr = Add(Literal(), Multiply(Lookup(matrix, matrix_indices), Lookup(vector, Id(j))))
    loop = For(length, Id(j), 1, expr)
    return loop

def sigmoid(x):
    return Divide(Literal(), Add(Literal(), Exp(Subtract(Literal(), x))))

def standard(n, r, d):

    var_suffix = str(tempvar[0])
    tempvar[0] += 1

    i = "i" + var_suffix
    j = "j" + var_suffix
    k = "k" + var_suffix

    k_loop = For(d, Id(k), 1, Add(Multiply(Lookup(Id("features"), [Id(i), Id(k)]), Lookup(Id("weights"), [Id(j), Id(k)])), Literal()))
    j_loop = For(n, Id(j), 1, k_loop)
    i_loop = For(r, Id(i), 1, j_loop)

    return cost(i_loop) / CLOCK_FREQUENCY

def sparse(n, r, d, sparsity):

    var_suffix = str(tempvar[0])
    tempvar[0] += 1

    i = "i" + var_suffix
    j = "j" + var_suffix
    k = "k" + var_suffix

    n_sparse = int(n*sparsity)

    else_branch = If(GreaterThan(Literal(), Literal()), Add(Literal(), Literal()), Add(Literal(), Literal()))
    else_branch.selectivity = 0.5

    condition = EqualTo(Lookup("idX", Id("i")), Lookup("idY", Id("j")))
    computation = Add(Multiply(Lookup("X", Id("i")), Lookup("Y", Id("j"))), Literal())
    branch = If(condition, computation, else_branch)
    branch.selectivity = sparsity / (2.0 - sparsity)
    branch2 = If(GreaterThan(Literal(), Literal()), Literal(), Literal())
    branch2.selectivity = 1.0
    loop1 = For(2 * n_sparse - (n_sparse * sparsity), Id("i"), 1, StructLiteral([branch, branch2, branch2]))

    loop_body2 = Add(Literal(), Literal())
    loop2 = For(2 * n_sparse, Id("i"), 1, loop_body2)

    j_loop = For(n, Id(j), 1, Let(Id("_"), loop1, loop2))
    i_loop = For(r, Id(i), 1, j_loop)

    branch = If(Lookup("weights", [Id("i"), Id("j")]), Let("sparse_weights[i]", Literal(), Literal()), Literal())
    branch.selectivity = sparsity
    j2_loop = For(d, Id("j"), 1, branch)
    i2_loop = For(r, Id("i"), 1, j2_loop)

    return (cost(i_loop) + 2 * cost(i2_loop)) / CLOCK_FREQUENCY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, dest='n', help="number of data points")
    parser.add_argument('-m', type=int, required=True, dest='d', help="dimensionality")
    parser.add_argument('-r', type=int, required=True, dest='r', help="Number of rates to test")
    parser.add_argument('-s', type=float, required=True, dest='s', help="Sparsity")

    # Optional flags
    parser.add_argument('-v', action="store_true", help="enable verbose output")
    parser.add_argument('-f', action="store_true", help="return time (using CLOCK_FREQUENCY)\
            instead of raw cost")
    args = parser.parse_args()

    n = args.n
    dims = args.d
    rates = args.r
    sparsity = args.s

    verbose = args.v
    use_frequency = args.f

    print "Matrix multiply:", cost(naive_matrix_multiply(Id("features"), Id("weights"), Id("results"), n, dims, rates)) / CLOCK_FREQUENCY
    print "Blocked matrix multiply:", cost(blocked_matrix_multiply(Id("features"), Id("weights"), Id("results"), n, dims, rates)) / CLOCK_FREQUENCY
    print "Sparse vectors:", sparse(n, rates, dims, sparsity)
    
if __name__=='__main__':
    main()
