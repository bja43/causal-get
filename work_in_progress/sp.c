

// FROM CHATGPT


#include <stdio.h>
#include <stdlib.h>


void print_perm(uint32_t *arr, size_t n);
void sjt_perm(uint32_t *arr, size_t n);
void heap_perm(uint32_t *arr, size_t n);



int main() {
    size_t n = 12;
    uint32_t *arr = malloc(n * sizeof(uint32_t));

    for (size_t i = 0; i < n; i++) arr[i] = i;

    // print_perm(arr, n);

    // printf("start\n");
    // sjt_perm(arr, n);
    // printf("end\n");

    // putchar('\n');

    printf("start\n");
    heap_perm(arr, n);
    printf("end\n");

    free(arr);
}


void print_perm(uint32_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%u ", arr[i]);
    putchar('\n');
}


void sjt_perm(uint32_t *arr, size_t n) {
    if (n == 0) return;

    // Direction: -1 = left, 1 = right, 0 = stationary
    int8_t *dirs = malloc(n * sizeof(int8_t));
    if (!dirs) return;

    // Initialization
    // All initially move left
    // 0 is immobile at start
    for (uint32_t i = 0; i < n; i++) dirs[i] = -1;
    dirs[0] = 0; 

    // print_perm(arr, n);

    while (1) {
        // Step 1: Find largest mobile integer
        int32_t i = -1;
        uint32_t x = 0;
        for (size_t j = 0; j < n; j++) {
            int8_t d = dirs[arr[j]];
            int32_t k = j + d;
            if (d != 0 && k >= 0 && k < n && arr[j] > arr[k]) {
                if (arr[j] > x) {
                    x = arr[j];
                    i = j;
                }
            }
        }

        if (i == -1) break; // Done

        int8_t d = dirs[x];
        int32_t k = i + d;

        // Step 2: Swap arr[i] and arr[k]
        uint32_t tmp = arr[i];
        arr[i] = arr[k];
        arr[k] = tmp;

        // Step 3: Update direction of x (now at position k)
        if (k == 0 || k == n - 1 || arr[k + dirs[arr[k]]] > x) {
            dirs[x] = 0;
        }

        // Step 4: Reverse directions of elements larger than x
        for (size_t j = 0; j < n; j++) {
            if (arr[j] > x) {
                dirs[arr[j]] = (j < k) ? 1 : -1;
            }
        }

        // print_perm(arr, n);
    }

    free(dirs);
}


void heap_perm(uint32_t *arr, size_t n) {
    if (n == 0) return;

    // Create a count array (like a stack frame counter)
    uint32_t *c = malloc(n * sizeof(uint32_t));
    if (!c) return;

    for (size_t i = 0; i < n; i++) c[i] = 0;

    // print_perm(arr, n); // First permutation

    size_t i = 0;
    while (i < n) {
        if (c[i] < i) {
            if (i % 2 == 0) {
                // swap(0, i)
                uint32_t tmp = arr[0];
                arr[0] = arr[i];
                arr[i] = tmp;
            } else {
                // swap(c[i], i)
                uint32_t tmp = arr[c[i]];
                arr[c[i]] = arr[i];
                arr[i] = tmp;
            }
            // print_perm(arr, n);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }

    free(c);
}

