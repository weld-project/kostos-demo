#ifdef __linux__
#define _BSD_SOURCE 500
#define _POSIX_C_SOURCE 2
#endif

#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>

#include <time.h>

/** Multiples an n x m matrix A with an m x n matrix B and stores the result in R.
 * A, B, and C are stored row-major.
 * 
 * @param A the left matrix
 * @param B the right matrix
 * @param R the result. Should be at least n * l * sizeof(float) in size.
 * @param n the number of rows in A.
 * @param m the number of columns in A, and the number of rows in B.
 * @param l the number of columns in B.
 * @param transpose_left use A^T instead of A. n and m are the sizes of the *transposed* A matrix.
 * @param transpose_right use B^T instead of B.
 */
typedef void (*matmul_func_t) (const void*,
        const void*,
        float *,
        int,
        int,
        int,
        int);

typedef enum {
    MATMUL_BLOCKED = 0, 
    MATMUL_NAIVE,
    MATMUL_MAX,
} matmul_kind_t;

static inline int min(int a, int b) {
    return a <= b ? a : b;
}

static inline int max(int a, int b) {
    return a >= b ? a : b;
}

static inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

struct sparse_vector {
    float *values;
    int *index;
};
typedef struct sparse_vector sparse_vector;

/* End-to-end workload:
- Perform some feature transformation (maybe group by?)
- Call train()
- Call test()
*/

float log_liklihood(const float *restrict features,
        const float *restrict target,
        const float *restrict weights,
        size_t n,
        size_t dims) {

    float ll = 0.0;
    for (int i = 0; i < n; i++) {
        float score = 0.0;
        for (int j = 0; j < dims; j++) {
            score += features[i*dims + j] * weights[j];
        }
        ll += target[i] * score - log(1 + exp(score));
    }
    return ll;
}

float dense_dot(const float *A,
        const float *B,
        int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (A[i] * B[i]);
    }
    return sum;
}

float sparse_dot(const sparse_vector *A,
        const sparse_vector *B,
        int n,
        int m) {
    int i = 0;
    int j = 0;
    float sum = 0.0;
    while (i < n && j < m) {
        int orig_i = A->index[i];
        int orig_j = B->index[j];
        if (orig_i == orig_j) {
            sum += (A->values[i] * B->values[j]);
            i++; j++;
        } else if (orig_i < orig_j) {
            i++;
        } else {
            j++;
        }
        if (i >= n) {
            break;
        }
        if (j >= m) {
            break;
        }
    }
    return sum;
}

void naive_matrix_multiply(const float *A,
        const float *B,
        float *R,
        int n,
        int m,
        int l) {

    memset(R, 0, sizeof(float) * n * l);
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            const float* A_row = A + (j * m);
            const float* B_row = B + (i * m);
            R[i*n + j] = sigmoid(dense_dot(A_row, B_row, m));
        }
    }

}

void blocked_matrix_multiply(const float *A,
        const float *B,
        float *R,
        int n,
        int m,
        int l) {

    memset(R, 0, sizeof(float) * n * l);
    const unsigned BS = 32;
    for (int i = 0; i < l; i+=BS) {
        for (int j = 0; j < n; j+=BS) {
            for (int k = 0; k < m; k+=BS) {
                for (int ii = i; ii < min(l, i + BS); ii++) {
                    for (int jj = j; jj < min(n, j + BS); jj++) {
                        const float* A_row = A + (jj * m);
                        const float* B_row = B + (ii * m);
                        R[ii*l + jj] += dense_dot(A_row, B_row, min(BS, m - k));
                        // for (int kk = k; kk < min(m, k + BS); kk++) {
                        //    R[ii*l + jj] += A[ii*m + kk] * B[m*jj + kk];
                        // }
                    }
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < l; j++) {
            R[i*l + j] = sigmoid(R[i*l + j]);
        }
    }

}

/** Print parameters to a string for reporting */
static char *stringify_params(size_t n, size_t dims, size_t num_rates, float sparsity) {
    static char buf[8192];
    snprintf(buf, sizeof(buf), "(n=%zu, dims=%zu, num_rates=%zu, sparsity=%f)",
            n, dims, num_rates, sparsity);
    return buf;
}

void bench(size_t n, size_t dims, size_t num_rates, size_t batchsize, float sparsity) {
    float *features = malloc(sizeof(float) * n * dims);
    float *labels = malloc(sizeof(float) * n);
    float *weights = malloc(sizeof(float) * num_rates * dims);
    float *results = malloc(sizeof(float) * n * num_rates);

    clock_t start, end;

    // Generate some garbage data.
    for (int i = 0; i < n; i++) {
        float label = (float)(rand() % 100) / 100.0;
        for (int j = 0; j < dims; j++) {
            float rand_number = (float) rand() / (float) RAND_MAX;
            if (rand_number <= sparsity) {
                features[i*dims + j] = label + j / 100.0;
            } else {
                features[i*dims + j] = 0.0;
            }
        }
        labels[i] = label;
    }

    for (int i = 0; i < num_rates; i++) {
        for (int j = 0; j < dims; j++) {
            float rand_number = (float) rand() / (float) RAND_MAX;
            if (rand_number <= sparsity) {
                weights[i*dims + j] = (float)(rand() % 100) / 100.0;
            } else {
                weights[i*dims + j] = 0.0;
            }
        }
    }
    
    // Run logistic regression invidivually on each learning rate.
    double total_time;

    start = clock();
    naive_matrix_multiply(features, weights, results, n, dims, num_rates); 
    end = clock();
    double total_time2 = ((double)end - (double)start) / CLOCKS_PER_SEC; 
    printf("%s: %f %s %f\n", "Matrix multiply", total_time2, stringify_params(n, dims, num_rates, sparsity), results[0]);

    start = clock();
    blocked_matrix_multiply(features, weights, results, n, dims, num_rates);
    end = clock();
    double total_time3 = ((double)end - (double)start) / CLOCKS_PER_SEC;
    printf("%s: %f %s %f\n", "Blocked matrix multiply", total_time3, stringify_params(n, dims, num_rates, sparsity), results[0]);

    start = clock();

    sparse_vector *sparse_features = malloc(sizeof(sparse_vector) * n);
    int *sparse_feature_lengths = malloc(sizeof(int) * n);
    float *feature_values = malloc(sizeof(float) * n * dims);
    int *feature_indexes = malloc(sizeof(int) * n * dims);

    sparse_vector *sparse_weights = malloc(sizeof(sparse_vector) * num_rates);
    int *sparse_weight_lengths = malloc(sizeof(int) * num_rates);
    float *weight_values = malloc(sizeof(float) * num_rates * dims);
    int *weight_indexes = malloc(sizeof(int) * num_rates * dims);

    for (int i = 0; i < n; i++) {
        sparse_feature_lengths[i] = 0;
        sparse_features[i].values = (feature_values + i*dims);
        sparse_features[i].index = (feature_indexes + i*dims);
        for (int j = 0; j < dims; j++) {
            if (features[i*dims + j] > 0.0) {
                sparse_features[i].values[sparse_feature_lengths[i]] = features[i*dims + j];
                sparse_features[i].index[sparse_feature_lengths[i]] = j;
                sparse_feature_lengths[i]++;
            }
        }
    }

    for (int i = 0; i < num_rates; i++) {
        sparse_weight_lengths[i] = 0;
        sparse_weights[i].values = (weight_values + i*dims);
        sparse_weights[i].index = (weight_indexes + i*dims);
        for (int j = 0; j < dims; j++) {
            if (weights[i*dims + j] > 0.0) {
                sparse_weights[i].values[sparse_weight_lengths[i]] = weights[i*dims + j];
                sparse_weights[i].index[sparse_weight_lengths[i]] = j;
                sparse_weight_lengths[i]++;
            }
        }
    }

     for (int i = 0; i < num_rates; i++) {
        for (int j = 0; j < n; j++) {
            results[i*n + j] = sigmoid(sparse_dot(sparse_weights + i, sparse_features + j, sparse_weight_lengths[i], sparse_feature_lengths[j]));
        }
    }
    end = clock();
    free(sparse_feature_lengths);
    free(feature_values);
    free(feature_indexes);
    free(sparse_weight_lengths);
    free(weight_values);
    free(weight_indexes);
    free(sparse_features);
    double total_time4 = ((double)end - (double)start) / CLOCKS_PER_SEC;
    printf("%s: %f %s %f\n", "Sparse vectors", total_time4, stringify_params(n, dims, num_rates, sparsity), results[0]);

    free(features);
    free(labels);
    free(weights);
    free(results);
}

void print_usage(char *prog) {
    fprintf(stderr, "%s -n <n> -m <dims> -r <num_reates> -b <batchsize> -k 0|1 (blocked|naive matrix) -s <sparsity>\n", prog);
}

int main(int argc, char **argv) {
    srand(12);

    size_t n = 10000;
    size_t dims = 1000;
    size_t num_rates = 50;
    size_t batchsize = n;
    float sparsity = 0.001;

    int set_bs = 0;

    matmul_kind_t matmul = MATMUL_BLOCKED;

    int ch;
    while ((ch = getopt(argc, argv, "n:m:r:b:s:k:")) != -1) {
        switch (ch) {
            case 'n':
                n = atoi(optarg);
                // By default, do full gradient descent.
                if (!set_bs) {
                    batchsize = n;
                }
                break;
            case 'm':
                dims = atoi(optarg);
                break;
            case 'r':
                num_rates = atoi(optarg);
                break;
            case 'b':
                batchsize = atoi(optarg);
                set_bs = 1;
                break;
            case 's':
                sparsity = atof(optarg);
                break;
            case 'k': {
                int x = atoi(optarg);
                assert (x < MATMUL_MAX);
                matmul = (matmul_kind_t)x;
                break;
                      }
            case '?':
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }

    // Sanity checks
    assert(n > 0);
    assert(batchsize > 0);
    assert(n >= batchsize);
    assert(num_rates > 0);
    assert(sparsity >= 0.0 && sparsity <= 1.0);

    // Run this to check correctness of the implementation.
    bench(n, dims, num_rates, batchsize, sparsity);
}
