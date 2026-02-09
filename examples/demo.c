#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void print_array_info(const Array *arr, const char *name) {
  printf("%s: shape=[", name);
  for (size_t i = 0; i < arr->ndim; i++) {
    printf("%zu", arr->shape[i]);
    if (i < arr->ndim - 1)
      printf(", ");
  }
  printf("], size=%zu, numc_type=%d, contiguous=%s\n", arr->size,
         arr->numc_type, array_is_contiguous(arr) ? "yes" : "no");
}

void demo_array_creation(void) {
  printf("\n=== Array Creation Demo ===\n\n");

  // Create zeros array
  printf("1. Creating 2x3 zeros array (INT):\n");
  size_t zeros_shape[] = {2, 3};
  Array *zeros = array_zeros(2, zeros_shape, NUMC_TYPE_INT);
  print_array_info(zeros, "zeros");
  array_print(zeros);
  array_free(zeros);

  // Create ones array
  printf("\n2. Creating 1x5 ones array (FLOAT):\n");
  size_t ones_shape[] = {5};
  Array *ones = array_ones(1, ones_shape, NUMC_TYPE_FLOAT);
  print_array_info(ones, "ones");
  array_print(ones);
  array_free(ones);

  // Create from data
  printf("\n3. Creating array from existing data:\n");
  int init_data[] = {1, 2, 3, 4, 5, 6};
  size_t arr_shape[] = {2, 3};
  ArrayCreate arr_create = {
      .ndim = 2,
      .shape = arr_shape,
      .numc_type = NUMC_TYPE_INT,
      .data = init_data,
      .owns_data = true,
  };
  Array *arr = array_create(&arr_create);
  print_array_info(arr, "arr");
  array_print(arr);
  array_free(arr);

  printf("\n4. Creating empty array:\n");
  // Create empty array
  ArrayCreate arr_create_empty = {
      .ndim = 2,
      .shape = arr_shape,
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  arr = array_empty(&arr_create_empty);
  print_array_info(arr, "arr");
  array_print(arr);
  array_free(arr);
}

void demo_array_operations(void) {
  printf("\n=== Array Operations Demo ===\n\n");

  // Element-wise math operations
  printf("1. Element-wise math operations:\n");
  int data_a[] = {10, 20, 30, 40, 50, 60};
  size_t shape_a[] = {6};
  ArrayCreate a_create = {
      .ndim = 1,
      .shape = shape_a,
      .numc_type = NUMC_TYPE_INT,
      .data = data_a,
      .owns_data = true,
  };
  Array *a = array_create(&a_create);
  printf("   a = ");
  array_print(a);

  int data_b[] = {1, 2, 3, 4, 5, 6};
  size_t shape_b[] = {6};
  ArrayCreate b_create = {
      .ndim = 1,
      .shape = shape_b,
      .numc_type = NUMC_TYPE_INT,
      .data = data_b,
      .owns_data = true,
  };
  Array *b = array_create(&b_create);
  printf("   b = ");
  array_print(b);

  // Pre-allocate output arrays for all operations
  ArrayCreate create = {
      .ndim = 1,
      .shape = shape_a,
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *sum = array_create(&create);
  Array *diff = array_create(&create);
  Array *prod = array_create(&create);
  Array *quot = array_create(&create);

  array_add(a, b, sum);
  printf("   a + b = ");
  array_print(sum);

  array_subtract(a, b, diff);
  printf("   a - b = ");
  array_print(diff);

  array_multiply(a, b, prod);
  printf("   a * b = ");
  array_print(prod);

  array_divide(a, b, quot);
  printf("   a / b = ");
  array_print(quot);

  array_free(a);
  array_free(b);
  array_free(sum);
  array_free(diff);
  array_free(prod);
  array_free(quot);

  // Array concatenation
  printf("\n2. Array concatenation (axis 0):\n");
  float data_x[] = {1.0f, 2.0f, 3.0f};
  size_t shape_x[] = {3};
  ArrayCreate x_create = {
      .ndim = 1,
      .shape = shape_x,
      .numc_type = NUMC_TYPE_FLOAT,
      .data = data_x,
      .owns_data = true,
  };
  Array *x = array_create(&x_create);
  printf("   x = ");
  array_print(x);

  float data_y[] = {4.0f, 5.0f};
  size_t shape_y[] = {2};
  ArrayCreate y_create = {
      .ndim = 1,
      .shape = shape_y,
      .numc_type = NUMC_TYPE_FLOAT,
      .data = data_y,
      .owns_data = true,
  };
  Array *y = array_create(&y_create);
  printf("   y = ");
  array_print(y);

  Array *concat = array_concatenate(x, y, 0);
  printf("   concat(x, y) = ");
  array_print(concat);

  array_free(x);
  array_free(y);
  array_free(concat);
}

void demo_array_slicing(void) {
  printf("\n=== Array Slicing Demo ===\n\n");

  // Create array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  size_t arr_shape[] = {10};
  Array *arr = array_zeros(1, arr_shape, NUMC_TYPE_INT);
  int *data = (int *)arr->data;
  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }
  printf("Original: ");
  array_print(arr);

  // Slice [2:8]
  size_t start1[] = {2};
  size_t stop1[] = {8};
  size_t step1[] = {1};
  Array *slice1 = array_slice(arr, start1, stop1, step1);
  printf("Slice [2:8]: ");
  array_print(slice1);

  // Slice [::2] (every other element)
  size_t start2[] = {0};
  size_t stop2[] = {10};
  size_t step2[] = {2};
  Array *slice2 = array_slice(arr, start2, stop2, step2);
  printf("Slice [::2]: ");
  array_print(slice2);

  // Slice [1::3] (start at 1, step by 3)
  size_t start3[] = {1};
  size_t stop3[] = {10};
  size_t step3[] = {3};
  Array *slice3 = array_slice(arr, start3, stop3, step3);
  printf("Slice [1::3]: ");
  array_print(slice3);

  array_free(arr);
  array_free(slice1);
  array_free(slice2);
  array_free(slice3);
}

void demo_array_flatten(void) {
  printf("\n=== Array Flatten Demo ===\n\n");

  // Create 1D array
  int data[][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  size_t shape[] = {3, 3};
  ArrayCreate arr_create = {
      .ndim = 2,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = false,
  };
  Array *arr = array_create(&arr_create);
  assert(arr);

  print_array_info(arr, "Array Flatten");
  printf("Original array: ");
  array_print(arr);
  // Flatten
  array_flatten(arr);
  printf("Flattened array: ");
  array_print(arr);

  array_free(arr);
}

void demo_array_reshape(void) {
  printf("\n=== Array Reshape Demo ===\n\n");

  // Create 1D array
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  size_t shape_1d[] = {12};
  ArrayCreate arr_create = {
      .ndim = 1,
      .shape = shape_1d,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&arr_create);
  printf("Original shape [12]:\n");
  print_array_info(arr, "arr");
  array_print(arr);

  // Reshape to 3x4
  printf("\nReshape to [3, 4]:\n");
  size_t shape_3x4[] = {3, 4};
  array_reshape(arr, 2, shape_3x4);
  print_array_info(arr, "arr");
  array_print(arr);

  // Reshape to 2x6
  printf("\nReshape to [2, 6]:\n");
  size_t shape_2x6[] = {2, 6};
  array_reshape(arr, 2, shape_2x6);
  print_array_info(arr, "arr");
  array_print(arr);

  array_free(arr);
}

void demo_array_copy(void) {
  printf("\n=== Array Copy & Views Demo ===\n\n");

  // Create original array
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {6};
  ArrayCreate arr_create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *original = array_create(&arr_create);
  printf("Original array:\n");
  print_array_info(original, "original");
  array_print(original);

  // Create a slice (view) - every other element
  size_t start[] = {0};
  size_t stop[] = {6};
  size_t step[] = {2};
  Array *slice = array_slice(original, start, stop, step);
  printf("\nSlice [::2] (view, non-contiguous):\n");
  print_array_info(slice, "slice");
  array_print(slice);

  // Copy the slice to make it contiguous
  Array *copied = array_copy(slice);
  printf("\nCopied array (now contiguous):\n");
  print_array_info(copied, "copied");
  array_print(copied);

  // Modify original - slice changes, but copy doesn't
  ((int *)original->data)[0] = 999;
  printf("\nAfter modifying original[0] to 999:\n");
  printf("  original: ");
  array_print(original);
  printf("  slice (shares data): ");
  array_print(slice);
  printf("  copied (independent): ");
  array_print(copied);

  array_free(original);
  array_free(slice);
  array_free(copied);
}

void demo_array_full(void) {
  printf("\n=== Array Fill Demo ===\n\n");

  // Create empty array and fill
  size_t shape[] = {8};
  Array *arr = array_zeros(1, shape, NUMC_TYPE_INT);
  printf("Zeros array: ");
  array_print(arr);

  // Fill with specific value using array_fill
  ArrayCreate src = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  int value = 42;
  Array *filled = array_full(&src, &value);
  printf("New array filled with 42: ");
  array_print(filled);

  array_free(arr);
  array_free(filled);
}

void demo_array_arange(void) {
  printf("\n=== Array Arrange Demo ===\n\n");

  // Create array with range [0, 10) step 1
  printf("1. array_arange(0, 10, 1, INT):\n");
  Array *arr1 = array_arange(0, 10, 1, NUMC_TYPE_INT);
  if (arr1) {
    printf("   Result: ");
    array_print(arr1);
    array_free(arr1);
  } else {
    printf("   Failed to create array\n");
  }

  // Create array with range [5, 20) step 2
  printf("\n2. array_arange(5, 20, 2, INT):\n");
  Array *arr2 = array_arange(5, 20, 2, NUMC_TYPE_INT);
  if (arr2) {
    printf("   Result: ");
    array_print(arr2);
    array_free(arr2);
  } else {
    printf("   Failed to create array\n");
  }

  // Create array with range [0, 100) step 10
  printf("\n3. array_arange(0, 100, 10, INT):\n");
  Array *arr3 = array_arange(0, 100, 10, NUMC_TYPE_INT);
  if (arr3) {
    printf("   Result: ");
    array_print(arr3);
    array_free(arr3);
  } else {
    printf("   Failed to create array\n");
  }

  // Create array with range [1, 6) step 1
  printf("\n4. array_arange(1, 6, 1, INT):\n");
  Array *arr4 = array_arange(1, 6, 1, NUMC_TYPE_INT);
  if (arr4) {
    printf("   Result: ");
    array_print(arr4);
    array_free(arr4);
  } else {
    printf("   Failed to create array\n");
  }

  // Test with FLOAT type
  printf("\n5. array_arange(0, 5, 1, FLOAT):\n");
  Array *arr5 = array_arange(0, 5, 1, NUMC_TYPE_FLOAT);
  if (arr5) {
    printf("   Result: ");
    array_print(arr5);
    array_free(arr5);
  } else {
    printf("   Failed to create array\n");
  }

  // Test with negative step
  printf("\n6. array_arange(10, 0, -1, INT) (negative step):\n");
  Array *arr6 = array_arange(10, 0, -1, NUMC_TYPE_INT);
  if (arr6) {
    printf("   Result: ");
    array_print(arr6);
    array_free(arr6);
  } else {
    printf("   Failed to create array\n");
  }

  // Test with negative step and larger interval
  printf("\n7. array_arange(20, 5, -2, INT) (negative step):\n");
  Array *arr7 = array_arange(20, 5, -2, NUMC_TYPE_INT);
  if (arr7) {
    printf("   Result: ");
    array_print(arr7);
    array_free(arr7);
  } else {
    printf("   Failed to create array\n");
  }

  // Test with range from 0 to negative
  printf("\n8. array_arange(0, -10, -1, INT) (0 to negative):\n");
  Array *arr8 = array_arange(0, -10, -1, NUMC_TYPE_INT);
  if (arr8) {
    printf("   Result: ");
    array_print(arr8);
    array_free(arr8);
  } else {
    printf("   Failed to create array\n");
  }

  // Test with range from negative to more negative
  printf("\n9. array_arange(-5, -15, -2, INT) (negative to more negative):\n");
  Array *arr9 = array_arange(-5, -15, -2, NUMC_TYPE_INT);
  if (arr9) {
    printf("   Result: ");
    array_print(arr9);
    array_free(arr9);
  } else {
    printf("   Failed to create array\n");
  }
}

void demo_array_linspace(void) {
  printf("\n=== Array Linspace Demo ===\n\n");

  // linspace(0, 10, 5) with FLOAT
  printf("1. array_linspace(0, 10, 5, FLOAT):\n");
  Array *arr1 = array_linspace(0, 10, 5, NUMC_TYPE_FLOAT);
  if (arr1) {
    printf("   Result: ");
    array_print(arr1);
    array_free(arr1);
  } else {
    printf("   Failed to create array\n");
  }

  // linspace(0, 100, 11) with INT
  printf("\n2. array_linspace(0, 100, 11, INT):\n");
  Array *arr2 = array_linspace(0, 100, 11, NUMC_TYPE_INT);
  if (arr2) {
    printf("   Result: ");
    array_print(arr2);
    array_free(arr2);
  } else {
    printf("   Failed to create array\n");
  }

  // linspace with single element
  printf("\n3. array_linspace(5, 5, 1, INT):\n");
  Array *arr3 = array_linspace(5, 5, 1, NUMC_TYPE_INT);
  if (arr3) {
    printf("   Result: ");
    array_print(arr3);
    array_free(arr3);
  } else {
    printf("   Failed to create array\n");
  }

  // linspace negative direction
  printf("\n4. array_linspace(10, 0, 5, FLOAT) (descending):\n");
  Array *arr4 = array_linspace(10, 0, 5, NUMC_TYPE_FLOAT);
  if (arr4) {
    printf("   Result: ");
    array_print(arr4);
    array_free(arr4);
  } else {
    printf("   Failed to create array\n");
  }

  // linspace with negative range
  printf("\n5. array_linspace(-10, 10, 5, FLOAT):\n");
  Array *arr5 = array_linspace(-10, 10, 5, NUMC_TYPE_FLOAT);
  if (arr5) {
    printf("   Result: ");
    array_print(arr5);
    array_free(arr5);
  } else {
    printf("   Failed to create array\n");
  }
}

void demo_array_astype(void) {
  printf("\n=== Array Type Conversion Demo ===\n\n");

  // Create integer array
  printf("1. Converting INT -> FLOAT:\n");
  int int_data[] = {10, 20, 30, 40, 50};
  size_t shape[] = {5};
  ArrayCreate int_create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = int_data,
      .owns_data = true,
  };
  Array *arr = array_create(&int_create);
  printf("   Original (INT): ");
  array_print(arr);
  print_array_info(arr, "   ");

  // Convert to float
  array_astype(arr, NUMC_TYPE_FLOAT);
  printf("   After astype(FLOAT): ");
  array_print(arr);
  print_array_info(arr, "   ");

  // Convert to double
  printf("\n2. Converting FLOAT -> DOUBLE:\n");
  array_astype(arr, NUMC_TYPE_DOUBLE);
  printf("   After astype(DOUBLE): ");
  array_print(arr);
  print_array_info(arr, "   ");

  // Convert back to int (narrowing conversion)
  printf("\n3. Converting DOUBLE -> SHORT (narrowing):\n");
  array_astype(arr, NUMC_TYPE_SHORT);
  printf("   After astype(SHORT): ");
  array_print(arr);
  print_array_info(arr, "   ");

  array_free(arr);

  // Demonstrate conversion with larger arrays
  printf("\n4. Type conversion with 2D array:\n");
  float float_data[] = {1.7f, 2.3f, 3.9f, 4.1f};
  size_t shape_2d[] = {2, 2};
  ArrayCreate float_create = {
      .ndim = 2,
      .shape = shape_2d,
      .numc_type = NUMC_TYPE_FLOAT,
      .data = float_data,
      .owns_data = true,
  };
  Array *arr2d = array_create(&float_create);
  printf("   Original (FLOAT, 2x2):\n");
  array_print(arr2d);

  array_astype(arr2d, NUMC_TYPE_INT);
  printf("   After astype(INT) - note truncation:\n");
  array_print(arr2d);

  array_free(arr2d);

  // Demonstrate unsigned conversion
  printf("\n5. Converting INT -> UINT (signed to unsigned):\n");
  int signed_data[] = {-5, -1, 0, 1, 5};
  ArrayCreate signed_create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = signed_data,
      .owns_data = true,
  };
  Array *signed_arr = array_create(&signed_create);
  printf("   Original (INT): ");
  array_print(signed_arr);

  array_astype(signed_arr, NUMC_TYPE_UINT);
  printf("   After astype(UINT): ");
  array_print(signed_arr);
  printf("   Note: Negative values wrap around (two's complement)\n");

  array_free(signed_arr);
}

void demo_array_equal(void) {
  printf("\n=== Array Equal Demo ===\n\n");

  // Create two identical arrays
  int data_a[] = {1, 2, 3, 4, 5};
  int data_b[] = {1, 2, 3, 4, 5};
  int data_c[] = {1, 0, 3, 0, 5};
  size_t shape[] = {5};

  ArrayCreate create_a = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data_a,
      .owns_data = true,
  };
  ArrayCreate create_b = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data_b,
      .owns_data = true,
  };
  ArrayCreate create_c = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data_c,
      .owns_data = true,
  };

  Array *a = array_create(&create_a);
  Array *b = array_create(&create_b);
  Array *c = array_create(&create_c);

  printf("1. Comparing identical arrays:\n");
  printf("   a = ");
  array_print(a);
  printf("   b = ");
  array_print(b);
  Array *eq1 = array_equal(a, b);
  printf("   array_equal(a, b) = ");
  array_print(eq1);

  printf("\n2. Comparing different arrays:\n");
  printf("   a = ");
  array_print(a);
  printf("   c = ");
  array_print(c);
  Array *eq2 = array_equal(a, c);
  printf("   array_equal(a, c) = ");
  array_print(eq2);

  array_free(a);
  array_free(b);
  array_free(c);
  array_free(eq1);
  array_free(eq2);
}

void demo_array_allclose(void) {
  printf("\n=== Array Allclose Demo ===\n\n");

  // Float arrays with rounding differences
  float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float data_b[] = {1.0f, 2.000001f, 3.0f, 4.000001f, 5.0f};
  float data_c[] = {1.0f, 2.1f, 3.0f, 4.0f, 5.0f};
  size_t shape[] = {5};

  ArrayCreate create_a = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true,
  };
  ArrayCreate create_b = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true,
  };
  ArrayCreate create_c = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data_c, .owns_data = true,
  };

  Array *a = array_create(&create_a);
  Array *b = array_create(&create_b);
  Array *c = array_create(&create_c);

  printf("1. Nearly identical arrays (rtol=1e-5, atol=1e-8):\n");
  printf("   a = ");
  array_print(a);
  printf("   b = ");
  array_print(b);
  int result1 = array_allclose(a, b, 1e-5, 1e-8);
  printf("   allclose(a, b) = %s\n", result1 ? "true" : "false");

  printf("\n2. Arrays with larger difference:\n");
  printf("   a = ");
  array_print(a);
  printf("   c = ");
  array_print(c);
  int result2 = array_allclose(a, c, 1e-5, 1e-8);
  printf("   allclose(a, c, rtol=1e-5) = %s\n", result2 ? "true" : "false");
  int result3 = array_allclose(a, c, 0.1, 0.0);
  printf("   allclose(a, c, rtol=0.1)  = %s\n", result3 ? "true" : "false");

  array_free(a);
  array_free(b);
  array_free(c);
}

void demo_array_prod(void) {
  printf("\n=== Array Product Demo ===\n\n");

  int data[] = {1, 2, 3, 4, 5};
  size_t shape[] = {5};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *arr = array_create(&create);
  printf("Array: ");
  array_print(arr);

  int result = 0;
  array_prod(arr, &result);
  printf("prod([1,2,3,4,5]) = %d\n", result);

  array_free(arr);
}

void demo_array_mean(void) {
  printf("\n=== Array Mean Demo ===\n\n");

  // Create 3x4 integer array
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  size_t shape[] = {3, 4};
  ArrayCreate create = {
      .ndim = 2,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);
  printf("Input array (3x4):\n");
  array_print(arr);

  // Mean along axis=0 (collapse rows → shape [4])
  printf("\n1. mean(axis=0) — collapse rows:\n");
  Array *m0 = array_mean_axis(arr, 0);
  if (m0) {
    printf("   Result: ");
    array_print(m0);
    array_free(m0);
  }

  // Mean along axis=1 (collapse columns → shape [3])
  printf("\n2. mean(axis=1) — collapse columns:\n");
  Array *m1 = array_mean_axis(arr, 1);
  if (m1) {
    printf("   Result: ");
    array_print(m1);
    array_free(m1);
  }

  array_free(arr);
}

void demo_array_std(void) {
  printf("\n=== Array Std Demo ===\n\n");

  int data[] = {2, 4, 4, 4, 5, 5, 7, 9};
  size_t shape[] = {8};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *arr = array_create(&create);
  printf("Array: ");
  array_print(arr);

  Array *s = array_std_axis(arr, 0);
  if (s) {
    printf("std([2,4,4,4,5,5,7,9]) = ");
    array_print(s);
    array_free(s);
  }

  array_free(arr);

  // 2D example
  int data2[] = {1, 5, 3, 7};
  size_t shape2[] = {2, 2};
  ArrayCreate create2 = {
      .ndim = 2, .shape = shape2,
      .numc_type = NUMC_TYPE_INT, .data = data2, .owns_data = true,
  };
  Array *arr2 = array_create(&create2);
  printf("\n2D array:\n");
  array_print(arr2);

  Array *s0 = array_std_axis(arr2, 0);
  if (s0) {
    printf("std(axis=0) = ");
    array_print(s0);
    array_free(s0);
  }

  Array *s1 = array_std_axis(arr2, 1);
  if (s1) {
    printf("std(axis=1) = ");
    array_print(s1);
    array_free(s1);
  }

  array_free(arr2);
}

void demo_performance(void) {
  printf("\n=== Performance Demo ===\n\n");

  // Test 1: Small arrays
  printf("Test 1: Adding two 1,000-element arrays\n");
  size_t small_shape[] = {1000};

  clock_t start = clock();
  Array *a1 = array_ones(1, small_shape, NUMC_TYPE_INT);
  Array *b1 = array_ones(1, small_shape, NUMC_TYPE_INT);
  ArrayCreate sum1_create = {
      .ndim = 1,
      .shape = small_shape,
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *sum1 = array_create(&sum1_create);
  clock_t create_time = clock();

  array_add(a1, b1, sum1);
  clock_t add_time = clock();

  double create_sec = (double)(create_time - start) / CLOCKS_PER_SEC;
  double add_sec = (double)(add_time - create_time) / CLOCKS_PER_SEC;
  double total_sec = (double)(add_time - start) / CLOCKS_PER_SEC;

  printf("  Array creation time: %.9f s\n", create_sec);
  printf("  Addition time: %.9f s\n", add_sec);
  printf("  Total time: %.9f s\n", total_sec);
  printf("  Result: all elements = %d (expected: 2)\n", ((int *)sum1->data)[0]);
  printf("  Array is contiguous: %s\n\n",
         array_is_contiguous(sum1) ? "yes (fast path used)" : "no");

  array_free(a1);
  array_free(b1);
  array_free(sum1);

  // Test 2: Larger arrays
  printf("Test 2: Adding two 100,000-element arrays\n");
  size_t medium_shape[] = {100000};

  start = clock();
  Array *a2 = array_ones(1, medium_shape, NUMC_TYPE_INT);
  Array *b2 = array_ones(1, medium_shape, NUMC_TYPE_INT);
  ArrayCreate sum2_create = {
      .ndim = 1,
      .shape = medium_shape,
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *sum2 = array_create(&sum2_create);
  create_time = clock();

  array_add(a2, b2, sum2);
  add_time = clock();

  double create_sec2 = (double)(create_time - start) / CLOCKS_PER_SEC;
  double add_sec2 = (double)(add_time - create_time) / CLOCKS_PER_SEC;
  double total_sec2 = (double)(add_time - start) / CLOCKS_PER_SEC;

  printf("  Array creation time: %.9f s\n", create_sec2);
  printf("  Addition time: %.9f s\n", add_sec2);
  printf("  Total time: %.9f s\n", total_sec2);
  printf("  Result: all elements = %d (expected: 2)\n", ((int *)sum2->data)[0]);
  printf("  Array is contiguous: %s\n\n",
         array_is_contiguous(sum2) ? "yes (fast path used)" : "no");

  array_free(a2);
  array_free(b2);
  array_free(sum2);

  // Test 3: Very large arrays
  printf("Test 3: Adding two 1,000,000-element arrays\n");
  size_t large_shape[] = {1000000};

  start = clock();
  Array *a3 = array_ones(1, large_shape, NUMC_TYPE_INT);
  Array *b3 = array_ones(1, large_shape, NUMC_TYPE_INT);
  ArrayCreate sum3_create = {
      .ndim = 1,
      .shape = large_shape,
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *sum3 = array_create(&sum3_create);
  create_time = clock();

  array_add(a3, b3, sum3);
  add_time = clock();

  double create_sec3 = (double)(create_time - start) / CLOCKS_PER_SEC;
  double add_sec3 = (double)(add_time - create_time) / CLOCKS_PER_SEC;
  double total_sec3 = (double)(add_time - start) / CLOCKS_PER_SEC;

  printf("  Array creation time: %.9f s\n", create_sec3);
  printf("  Addition time: %.9f s\n", add_sec3);
  printf("  Total time: %.9f s\n", total_sec3);
  printf("  Result: all elements = %d (expected: 2)\n", ((int *)sum3->data)[0]);
  printf("  Array is contiguous: %s\n",
         array_is_contiguous(sum3) ? "yes (fast path used)" : "no");

  array_free(a3);
  array_free(b3);
  array_free(sum3);
}

int main(void) {
  printf("\n");
  printf("╔══════════════════════════════════════════════════════════╗\n");
  printf("║         NUMC - N-Dimensional Array Library Demo          ║\n");
  printf("║              NumPy-like Operations in C                  ║\n");
  printf("╚══════════════════════════════════════════════════════════╝\n");

  demo_array_creation();
  demo_array_operations();
  demo_array_slicing();
  demo_array_reshape();
  demo_array_flatten();
  demo_array_copy();
  demo_array_full();
  demo_array_arange();
  demo_array_linspace();
  demo_array_astype();
  demo_array_equal();
  demo_array_allclose();
  demo_array_prod();
  demo_array_mean();
  demo_array_std();
  demo_performance();

  printf("\n");
  printf("╔══════════════════════════════════════════════════════════╗\n");
  printf("║                     Demo Complete!                       ║\n");
  printf("║                                                          ║\n");
  printf("║  Features Demonstrated:                                  ║\n");
  printf("║  ✓ Array creation (zeros, ones, from data)               ║\n");
  printf("║  ✓ Element-wise operations (add, sub, mul, div)          ║\n");
  printf("║  ✓ Concatenation                                         ║\n");
  printf("║  ✓ Slicing & views                                       ║\n");
  printf("║  ✓ Reshaping                                             ║\n");
  printf("║  ✓ Flatten                                               ║\n");
  printf("║  ✓ Copying (contiguous conversion)                       ║\n");
  printf("║  ✓ Fill operations                                       ║\n");
  printf("║  ✓ Arrange (range generation)                            ║\n");
  printf("║  ✓ Linspace (linear spacing)                             ║\n");
  printf("║  ✓ Type conversion (astype - 100 type pairs)             ║\n");
  printf("║  ✓ Element-wise equality comparison                      ║\n");
  printf("║  ✓ Tolerance-based comparison (allclose)                  ║\n");
  printf("║  ✓ Product reduction                                      ║\n");
  printf("║  ✓ Mean along axis (reduction)                           ║\n");
  printf("║  ✓ Standard deviation along axis                         ║\n");
  printf("║  ✓ Optimized performance (type-specific kernels)         ║\n");
  printf("╚══════════════════════════════════════════════════════════╝\n");
  printf("\n");

  return 0;
}
