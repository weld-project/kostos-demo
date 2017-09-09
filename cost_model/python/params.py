# Machine parameters/constants for the CBO.

# ************************* System/CPU Parameters  *************************

"""
Instructions (Linux):

    Cores: lscpu
    Clock Frequency: lscpu
"""

# Number of CPU cores.
CORES = 4

# 2.9 GHz
CLOCK_FREQUENCY = 2E9


# ************************* Memory Parameters  *************************

"""
Instructions (Linux):

    L1, L2, L3 Cache Size: lscpu, divide by line size
    Clock Frequency: lscpu
    Cache Line Size: cat /proc/cpuinfo, read the clflush size field
"""

# Memory throughput at different levels of the heirarchy (index 0 is L1 cache,
# etc.).
L1_THROUGHPUT   = 529E9
L2_THROUGHPUT   = 350E9
L3_THROUGHPUT   = 120E9
MEM_THROUGHPUT  = 55E9

# Cache sizes in *blocks/lines*
L1_SIZE         = 500
L2_SIZE         = 4000
L3_SIZE         = 240000 #720000 

# Cache line size in bytes.
CACHE_LINE_SIZE = 64

# Memory Latencies
L1_LATENCY = 1
L2_LATENCY = 7
L3_LATENCY = 19
MEM_LATENCY = 36

# ************************* Instruction Parameters *************************

# Latency of a standard binary op (+ - / * >= etc.)
BINOP_LATENCY = 1
# Latency of a vectorized binary op (+ - / * >= etc.)
BINOP_VEC_LATENCY = 4

# Latency of a standard unary op (square / square root)
UNOP_LATENCY = 4
UNOP_VEC_LATENCY = 8

# Latency of an atomic add with no contention.
ATOMICADD_LATENCY = 20.0
# Penalty of contention for an atomic add.
ATOMICADD_PENALTY = 2000.0

# Given a constant condition, The number of branches fall a certain way for the
# branch predictor to predict it correctly subsequently.
BRANCHPRED_PREDICTABLE_IT_DIST = 10

# The max latency of a branch misprediction.
BRANCHPRED_LATENCY = 20
# The latency of executing a branching instruction.
BRANCH_LATENCY = 1

# The branch misprediction penalty. The penalty is an expected value; when
# selectivity is closer to 0.5, the penalty is closer to the full
# BRANCHPRED_LATENCY.
def BRANCH_MISPREDICT_PENALTY(selectivity):
    # Returns a parabola whose maxima is at at s=0.5
    return -2 * BRANCHPRED_LATENCY * abs(selectivity - 0.5) + BRANCHPRED_LATENCY + 3


# ************************* Hashtable Parameters *************************

INSERT_LATENCY_SINGLE = 4.5e-07 * CLOCK_FREQUENCY
INSERT_LATENCY_CONTEND = 1.1e-06 * CLOCK_FREQUENCY ## 4 threads ##
LOOKUP_LATENCY = 5.5e-07 * CLOCK_FREQUENCY
LOOKUP_LOCAL_LATENCY = 5.5e-08 * CLOCK_FREQUENCY
