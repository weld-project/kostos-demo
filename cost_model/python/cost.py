# Routines to get various cost metrics from an expression.
"""
The cost framework works as follows.

Costs are only computed for parallel constructs in the language, namely For
loops. Expressions not nested under for loops are not considered (this may
change later, e.g. if an "if" statement branches into two different
for loops).

Costs are decomposed into two components: a _memory cost_ and a
_processing cost_.

The processing cost states the amount of CPU used in a given loop. For example,
in the loop for(V, |x| -> x + 1), the total processing cost would be len(V),
since len(V) iterations of the loop are run and each iteration performs
(roughly) one cycle of processing. The processing cost assumes value are already
in registers.

Processing cost can also capture instruction-level optimizations such as
vectorization. For example, if we were to vectorize the above loop to have
stride 4 (for(V, 4, |x| -> x + (1,1,1,1))), the cost would become len(V) / 4
since each instruction still only takes one cycle, but 1/4 the number of
iterations are run.

Memory cost captures the cost of loading values from the memory hierarchy. The
primary aspect captured in the memory model is prefetching. The model computes,
based on selectivity, the cost of moving data into the L1 cache and overlaps
costs of fetching data into LLC (and eventually, L2) with the processing cost
to capture adjacent cache line prefetchers of modern CPUs.

TODO : Multi-core.
"""

from expressions import *
from extended_cost_model import *

def cost(expr, block_sizes, latencies):
    # Return the cost of an expression.

    # Only consider costs of loops.
    if not isinstance(expr, For):
        return 0

    # Get the processing cost - this is the cost of running each iteration
    # N times, where N is the number of iterations in the loop. This ignores
    # costs related to the memory heirarchy and assumes all values are loaded
    # into registers.
    p_cost = expr.cost({})

    def _get_lookups(expr, lookups=set(), loopIndices=[]):
        # Find Lookup (i.e. memory access) nodes in the expression tree.
        for c in expr.children():
            # Record the loop index (tracks nested loops).
            if isinstance(c, For):
                loopIndices.append(c.loopIdx)
            # Here, we'll annotate the Lookup nodes with a bunch of excess data
            # (e.g., is it sequential, etc.). This is hacky since we're adding
            # new fields to the object.
            if isinstance(c, Lookup):
                c.sequential = is_sequential(c, loopIndices)
                lookups.add(c)
            _get_lookups(c, lookups, loopIndices)
        return lookups

    # The list of lookups.
    lookups = list(_get_lookups(expr))
    # The cost for a given lookup for the L1 and L2 cache.
    lookup_costs = []
    # The cost for a given lookup for the L3 cache.
    l3_seq_costs = []

    for l in lookups:
        # Compute the cost of L1 access time. All elements do this access.
        # Assume an element on a line is loaded "at once."
        #l1_cost = expr.iters * expr.stride * 4 * latencies[0]

        # TODO check if the lookup is sequential.

        # Compute the cost of missing in L1.  This computes the probability of
        # accessing a cache line, and then weights the number of accessed
        # blocks (i.e., the total data size divded by the block size) by this
        # probability and the L2 access latency.
        l1_cost = sequential_cost(l.p_execute, 4 * expr.stride,
                                  expr.iters / expr.stride, block_sizes[0], latencies[1])
        l1_cost += random_cost(l.p_execute, 4 * expr.stride,
                               expr.iters / expr.stride, block_sizes[0], latencies[1])

        # Compute the cost of missing in L2 (same as description in L1).
        # TODO we're ignoring this now, assuming an L3 miss loads data into L2
        # as well/L2 data accesses are always prefetched from L3 and the
        # prefetch time is less than the processing + L1 access time). This
        # needs some refinement.
        l2_cost = sequential_cost(l.p_execute, 4 * expr.stride,
            expr.iters / expr.stride, block_sizes[1], latencies[2])
        l2_cost += random_cost(l.p_execute, 4 * expr.stride,
                expr.iters / expr.stride, block_sizes[1], latencies[2])

        # Compute the cost of missing in L3. An L3 miss which is sequential
        # is overlapped with the L1/processing cost. A random miss goes to
        # memory.
        l3_seq_cost = sequential_cost(l.p_execute, 4 * expr.stride,\
                expr.iters / expr.stride, block_sizes[2], latencies[3])
        l3_rnd_cost = random_cost(l.p_execute, 4 * expr.stride,\
                expr.iters / expr.stride, block_sizes[2], latencies[3])

        # Register loading cost (i.e. L1 access cost for each data element).
        # TODO probably an overestimate.
        reg_cost = (expr.iters * 4 * expr.stride / 8) * l.p_execute

        lookup_costs.append(l1_cost + l3_rnd_cost + reg_cost)
        l3_seq_costs.append(l3_seq_cost)

    # Get the highest L3 sequential cost. We assume all the prefetched L3 lines
    # are prefetched in parallel.
    # TODO bandwidth based penalty for more loads at once.
    highest_l3_cost = max(l3_seq_costs)
    # Sum the L1 and L2 lookup costs.
    mem_cost = sum(lookup_costs)

    # Get the effective L3 cost. Here, we check if the L3 cost
    # is masked by the prefetcher by "overlapping" it with the L1/L2
    # load costs and the processing cost.
    l3_cost = max(0.0, highest_l3_cost - (p_cost + mem_cost))

    # Return the sum of all the costs.
    return l3_cost + p_cost + mem_cost

def is_sequential(lookup, indices):
    # given a lookup expression and an ordered list of loop
    # indices (ordered by nesting), returns whether an access pattern
    # is sequential.
    #
    # Caveat: Only consider simple arithmetic computations right now
    # (i.e., only Add, Subtract, Multiply and Divide Expr nodes are allowed in the
    # index computation).
    #
    # TODO this should later be augmented to return the "distance" of
    # reuse between two accesses (e.g. A[j][i] has a reuse distance of len(j))

    # None lookup nodes designate implicit sequential access.
    if lookup is None:
        return True

    index = lookup.index[-1]
    def contains(node, f):
        if f == node:
            return True
        for c in node.children():
            if contains(c, f):
                return True
        return False

    # If the last loop index is accessed non-sequentially, the access is
    # not sequential.
    if isinstance(index, Multiply) or isinstance(index, Divide):
        if contains(index, indices[-1]):
            return False

    # Don't support complex expressions for now.
    # TODO this needs to be slightly more complex (e.g. Mods are special).
    if isinstance(index, BinaryExpr):
        if not isinstance(index, Add) and not isinstance(index, Subtract):
            return False

    for c in index.children():
        if not is_sequential(c, indicies):
            return False
    return True

