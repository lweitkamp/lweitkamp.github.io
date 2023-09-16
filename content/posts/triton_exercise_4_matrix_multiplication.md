---
title: "Triton Exercise 4: Matrix Multiplication"
date: 2023-08-05
draft: true
---



# Introduction
<!-- {{< figure src="/img/triton/matmul.svg" caption="Fig 1. A program takes a row of matrix A (red) and multiplies it with a column of matrix B (green). The result is a single value in the output matrix C." >}} -->
Remembering that data loading and storing from and to SRAM is expensive, we should look at the expenses here. We load one row of data from matrix A of K entries and we multiply it by 6 times K entries loaded from matrix B. The result is a single row of data in matrix C of 6 entries. that's 12 + 6K entries total.
> todo: math here needs some work.


# Blocked Matrix Multiplication
If we are serious about utilizing the cache more efficiently, we can instead chunk the matrices A and B into bock of rows and columns respectively. Lets set the block size for dividing up N to B_N = 2, and the block size for dividing up M to B_M = 2:

{{< figure src="/img/triton/blockify.svg" caption="Fig 2. We will work on blocks of the original matrices. For matrix A (red) we work on blocks of 2 rows each, for matrix B (green) we work on blocks of 2 columns each." >}}

Each program is then responsible for generating a 2x2 block of values for the output matrix C:

{{< figure src="/img/triton/blocked_matmul.svg" caption="Fig 3. Each program ID takes a block of columns of matrix A (red) and a block of rows of matrix B (green) and matrix-multiply them along the inner (K) dimension. The result will be blocks of the output matrix C." >}}

So let's compare this data loading / storing to the previous 'naive' approach. Here we load two rows of matrix A, a total of 2 times K entries. We load two columns of B, two times K entries. After multiplying, we have a 2x2 result that we store in the output matrix C. That's 4 + 4K entries total, a huge improvement over the naive approach.

> todo: math here needs some work.

## Individal Program
How will a program generate this 2x2 block of values? Let's zoom in on one specific program (program ID = 10). We have a few ways of accomplishing the output in the output matric C. If SRAM permits, we could simply matrix multiply the entire block of A with the entire block of B.

In the general case (large matrices) this is not possible, so we again divide the work up into blocks along the inner (K) dimension. Since this is the reduction dimension, we can create chunks along K for both blocks of A and B and multiply these together whilst accumulating the individual results. This is illustrated in the figure below:

{{< figure src="/img/triton/matmul_single_program.svg" caption="Fig 4. A program takes a block of B_N rows of matrix A (red) and a block of B_M columns of matrix B (green). These blocks are in turn chunked into blocks along the inner dimension (K) of size B_K and each individual bock will be matrix-multiplied (denoted with the @ sign). The results of each individual matrix multiplication will be accumulated and finally stored to its place in output matrix C." >}}

You can think of the matrix multiplication explained in the introduction as a special case of the blocked matrix multiplication, where B_K = K, B_N = 1 and B_M = M.

# The Assignment
So, how can we accomplish this in Triton?

- Pointer arithmetic
- We can load individual parts of the blocks of A and B into SRAM using `tl.load(BLOCK_POINTER)`. When using block pointers. We do need to be careful with out-of-memory accesses, and this can be ensured by providing the correct values in the `boundary_check` argument.
- In Triton, a matrix multiplication can be done using `tl.dot`, and accumulation can be done with `+=`, just like how you would expect it to work in Python.
- When work on the block is done, the output can be stored using `tl.store(BLOCK_POINTER)`, again ensuring the correct values are stored using `boundary_check`.