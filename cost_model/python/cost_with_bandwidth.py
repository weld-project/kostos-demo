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
since each instruction still only takes one cycle, but 1/4 the number of iterations
are run.

Memory cost captures the cost of loading values from the memory heirarchy. This
is an attempt at simplifying the model used before, making sure to take into
account the bandwidth of the memory sub-system.

TODO : Multi-core.
"""

from expressions import *
from extended_cost_model import *

import params

def cost(expr):
    # Return the cost of an expression.

    # Get the processing cost - this is the cost of running each iteration
    # N times, where N is the number of iterations in the loop. This ignores
    # costs related to the memory heirarchy and assumes all values are loaded
    # into registers.
    p_cost = expr.cost({})

    # Only consider costs of loops.
    if not isinstance(expr, For):
        return p_cost

    # CPU clock frequency.
    clock_frequency = params.CLOCK_FREQUENCY

    # Memory throughput at different levels of the heirarchy (index 0 is L1 cache, etc.).
    memory_throughput = [params.L1_THROUGHPUT, params.L2_THROUGHPUT, params.L3_THROUGHPUT, params.MEM_THROUGHPUT]
    # Cache sizes in terms of blocks (cache size at level i in bytes = cache_size[i] * block_size).
    cache_sizes = [params.L1_SIZE, params.L2_SIZE, params.L3_SIZE]

    # Cache block (line) size at different levels of the memory hierarchy.
    block_size = params.CACHE_LINE_SIZE
    # Memory access latencies at different levels of the memory hierarchy.
    latencies = [params.L1_LATENCY, params.L2_LATENCY, params.L3_LATENCY, params.MEM_LATENCY]

    def _get_lookups(expr, lookups=set()):
        # Find Lookup (i.e. memory access) nodes in the expression tree.
        for c in expr.children():
            if isinstance(c, Lookup):
                lookups.add(c)
            _get_lookups(c, lookups)
        return lookups

    # The list of lookups.
    lookups = list(_get_lookups(expr))
    m_cost = 0

    for l in lookups:
        num_lookups = (l.loops * l.p_execute)
        l_reuse_distance = reuse_distance(l, lookups, l.loops_seq, block_size)

        if l.sequential:
            # Sequential access - use bandwidth.
            mem_lookups = (num_lookups * l.elemSize)

            cache_level = len(cache_sizes)
            for i in xrange(len(cache_sizes)):
                if cache_sizes[i] > l_reuse_distance:
                    cache_level = i
                    break

            avg_mem_throughput = memory_throughput[cache_level]
            if not get_is_sequential([l.loops_seq[-1]], l.index):
                avg_mem_throughput /= float(params.CACHE_LINE_SIZE / l.elemSize)
            m_cost += (((mem_lookups) / avg_mem_throughput) * clock_frequency)
        else:
            # Random access - use latency.
            rand_cost = 0.0
            vector_size = l.vector.length * l.elemSize
            prev_p = 0.0
            for i in xrange(len(cache_sizes)):
                # Number of blocks in the vector.
                blocks = vector_size / block_size
                if i == len(cache_sizes) - 1:
                    blocks *= params.CORES
                p = cache_sizes[i] / blocks
                p = min(1.0, max(p, 0.0))
                old_p = p
                p -= prev_p
                prev_p += p
                rand_cost += p * latencies[i]
                if old_p == 1.0:
                    break

            # Factor in the DRAM access latency for "unaccounted" probabilities.
            if prev_p != 1.0:
                p = 1.0 - prev_p
                rand_cost += p * latencies[-1]
            m_cost += (num_lookups * rand_cost)

    # Add memory and processing cost here.
    # TODO what's the correct way to combine these?
    return p_cost + m_cost
