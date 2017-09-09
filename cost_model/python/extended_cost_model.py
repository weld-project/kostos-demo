# Routines implementing the extended cost model.

def P_seq_cache_miss(s, block_size):
    # Returns probability of a cache miss which is sequential.
    return pow(1. - pow(1.0 - s, block_size), 2)

def P_rnd_cache_miss(s, block_size):
    # Returns the probability of a cache miss which is random.
    t = pow(1. - s, block_size)
    return t - pow(t, 2)

def num_cache_misses(width, length, block_size, prob):
    return prob * (width * length) / block_size

def sequential_cost(s, dwidth, dlength, block_size, latency):
    # Returns the cost of sequential misses.
    p_seq = P_seq_cache_miss(s, block_size)
    num_seq_misses = num_cache_misses(dwidth, dlength, block_size, p_seq)
    return num_seq_misses * latency

def random_cost(s, dwidth, dlength, block_size, latency):
    # Returns the cost of random misses.
    p_rnd = P_rnd_cache_miss(s, block_size)
    num_rnd_misses = num_cache_misses(dwidth, dlength, block_size, p_rnd)
    return num_rnd_misses * latency
