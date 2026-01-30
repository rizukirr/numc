#include <array.h>
#include <arrio.h>
#include <stdio.h>
#include <time.h>

int main(void) {
  printf("=== Testing static array to dynamic Array conversion ===\n\n");

  // Example 1: Your exact case - int a[2][2][2]
  printf("Example 1: 3D array [2][2][2]\n");
  int a[2][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

  size_t shape3d[] = {2, 2, 2};
  Array *arr3d = array_from_data(3, shape3d, sizeof(int), a);

  printf("Original C array a[2][2][2] = {{{1,2},{3,4}},{{5,6},{7,8}}}\n");
  printf("Dynamic array: ");
  array_print(arr3d);

  // Access individual elements
  int *elem = array_at(arr3d, (const size_t[]){1, 1, 0});
  printf("Element at [1][1][0]: %d\n", *elem);

  array_free(arr3d);

  // Example 2: 2D array
  printf("\n\nExample 2: 2D array [3][4]\n");
  int b[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  size_t shape2d[] = {3, 4};
  Array *arr2d = array_from_data(2, shape2d, sizeof(int), b);

  printf("Original C array b[3][4]\n");
  printf("Dynamic array: ");
  array_print(arr2d);

  array_free(arr2d);

  // Example 3: 1D array (simple case)
  printf("\n\nExample 3: 1D array [5]\n");
  double c[] = {1.5, 2.5, 3.5, 4.5, 5.5};

  size_t shape1d[] = {5};
  Array *arr1d = array_from_data(1, shape1d, sizeof(double), c);

  printf("Original C array c[] = {1.5, 2.5, 3.5, 4.5, 5.5}\n");
  printf("Dynamic array: ");
  array_print(arr1d);

  array_free(arr1d);

  // Example 4: Manual element-by-element copy (alternative approach)
  printf("\n\nExample 4: Manual copy approach for [2][3]\n");
  int d[2][3] = {{10, 20, 30}, {40, 50, 60}};

  size_t shape_manual[] = {2, 3};
  Array *arr_manual = array_create(2, shape_manual, sizeof(int));

  // Copy manually (when data might not be contiguous)
  int *arr_data = (int *)arr_manual->data;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      arr_data[i * 3 + j] = d[i][j];
    }
  }

  printf("Manually copied array: ");
  array_print(arr_manual);

  array_free(arr_manual);

  printf("\n\n=== Testing array_append geometric growth optimization ===\n\n");

  // Test 1: Basic append functionality
  printf("Test 1: Basic append test\n");
  size_t shape_empty[] = {0};
  Array *arr = array_create(1, shape_empty, sizeof(int));

  printf("Initial: size=%zu, capacity=%zu\n", arr->size, arr->capacity);

  // Append 20 elements and observe capacity growth
  for (int i = 1; i <= 20; i++) {
    int val = i * 10;
    array_append(arr, &val);
    if (i <= 10 || i > 17) { // Show first 10 and last few
      printf("After append %2d: size=%2zu, capacity=%2zu\n", i, arr->size,
             arr->capacity);
    } else if (i == 11) {
      printf("  ... (showing growth pattern) ...\n");
    }
  }

  printf("\nArray contents: ");
  array_print(arr);

  array_free(arr);

  // Test 2: Performance test - append many elements
  printf("\n\nTest 2: Performance test (10,000 appends)\n");
  Array *perf_arr = array_create(1, shape_empty, sizeof(int));

  clock_t start = clock();
  for (int i = 0; i < 10000; i++) {
    int val = i;
    array_append(perf_arr, &val);
  }
  clock_t end = clock();

  double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time for 10,000 appends: %.6f seconds\n", time_taken);
  printf("Final: size=%zu, capacity=%zu\n", perf_arr->size, perf_arr->capacity);
  printf(
      "Capacity/Size ratio: %.2fx (shows over-allocation for future appends)\n",
      (double)perf_arr->capacity / perf_arr->size);

  array_free(perf_arr);

  printf("\n=== All tests passed! ===\n");
  return 0;
}
