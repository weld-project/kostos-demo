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
#include <sys/time.h>

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

struct sparse_vector {
    float *values;
    int *index;
};
typedef struct sparse_vector sparse_vector;

void print_usage(char *prog) {
    fprintf(stderr, "%s -n <n> -m <dims> -r <num_reates> -b <batchsize> -k 0|1 (blocked|naive matrix) -s <sparsity>\n", prog);
}

float measure_sparsity(float *features, int k, int dims, int n) {
  float s_approx = 0.0;
  for ( int i = 0; i < k; i++ ) {
    if ( features[rand() % (dims*n)] == 0 ) {
      s_approx += 1;
    }
  }

  return 1 - (s_approx / k);
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

    int k = 1000; /* number of samples to take when approximating sparsity */
    char* param_flag;

    int ch;
    while ((ch = getopt(argc, argv, "n:m:r:b:s:k:t:")) != -1) {
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
            case 't':
              param_flag = optarg;
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

    float *labels = malloc(sizeof(float) * n);
    float *features = malloc(sizeof(float) * n * dims);
    // Run this to check correctness of the implementation.
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
    
    struct timeval start, end, diff;

    switch (param_flag[0]) {
    case 's': {
      gettimeofday(&start, 0);
      float sp = measure_sparsity(features, k, dims, n);
      gettimeofday(&end, 0);
      timersub(&end, &start, &diff);
      printf("Sparsity: time=%ld.%06ld result=%f\n", (long) diff.tv_sec, (long) diff.tv_usec,
             sp);
      break;
    }
    }

    return 0;
}
