# RV-Sparse Coding Challenge

## Overview

This repository contains my solution to the coding challenge for the [RV-Sparse LFX Mentorship Program](https://mentorship.lfx.linuxfoundation.org/project/deb25137-a736-4f6f-a673-a02e52e55758).

The goal of the project is to develop a lightweight C library that implements key sparse linear algebra primitives particularly sparse matrix multiplication using RISC-V vector intrinsics.

---

## What I Implemented

The challenge required implementing the `sparse_multiply` function in `challenge.c`,
which performs the following tasks with **zero dynamic memory allocation**:

1. Convert a row-major matrix `A` into **Compressed Sparse Row (CSR)** format,
2. Perform **matrix-vector multiplication** (`y = A * x`) using the CSR representation.

```c
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
);
```

---

### Sparse Matrix
A **sparse matrix** is a matrix in which most of its elements are zero. When operating on sparse matrices, using the dense-matrix structures and and algorithms are slow and inefficient as processing and memory are wasted on zeros. So it is beneficial to use specialized algorithms and data structures that take advantage of the sparse structure of the matrix. There are various such storage solutions for sparse matrices like,

- DOK (Dictionary of keys), LIL (List of lists), COO (Coordinate list)
    - Typically used to construct the matrices.
- CSR (Compressed Sparse Row), CSC (Compressed Sparse Column)
    - Support efficient access and matrix operations


#### CSR Format

Compressed Sparse Row (CSR) is a standard format for storing sparse matrices efficiently which enables **fast row access and matrix-vector multiplications**.
Instead of storing all `rows × cols` elements, it stores only the non-zero values using
three arrays:

| Array |  Size | Description |
|---|---|---|
| `values[]` | `nnz` | Contains all the non-zero values in row-major ordering. |
| `col_indices[]` | `nnz` | Contains the column indices of corresponding elements in the `values[]` array.|
| `row_ptrs[]` | `m+1` | Contains the index in `values[]` and `col_indices[]` where the given row starts. |

> **nnz:** number of non-zero elements in the matrix.<br>
**m:** number of rows in the matrix.

##### row_ptrs[]
The row offsets are calculated using the following recursive formula.
$$\text{row\_ptrs}[0] = 0$$
$$\text{row\_ptrs}[j+1] = \text{row\_ptrs}[j] + \text{nnz}(\text{row}_j)$$

> where $\text{nnz}(\text{row}_k)$ is the number of non-zero elements in the $k$-th row.

#### Example
$$
A = \begin{bmatrix}
1 & 0 & 2 & 0 \\
0 & 3 & 0 & 0 \\
0 & 0 & 0 & 0 \\
4 & 5 & 0 & 0 \\
0 & 6 & 7 & 8
\end{bmatrix}
\quad
\begin{aligned}
&\text{values}       = [1, 2, 3, 4, 5, 6, 7, 8] \\
&\text{col\_indices} = [0, 2, 1, 0, 1, 1, 2, 3] \\
&\text{row\_ptrs}    = [0, 2, 3, 3, 5, 8]
\end{aligned}
$$

### Matrix-Vector Multiplication with CSR


$$
A \cdot x =
\begin{bmatrix}
1 & 0 & 2 & 0 \\
0 & 3 & 0 & 0 \\
0 & 0 & 0 & 0 \\
4 & 5 & 0 & 0 \\
0 & 6 & 7 & 8
\end{bmatrix}
\begin{bmatrix} 3 \\ 5 \\ 2 \\ 1 \end{bmatrix}
= \begin{bmatrix}
1 \times 3 + 2 \times 2 \\
3 \times 5 \\
0 \\
4 \times 3 + 5 \times 5 \\
6 \times 5 + 7 \times 2 + 8 \times 1
\end{bmatrix}
$$

Instead of iterating over all elements (including zeros), CSR lets us compute
only the non-zero contributions.

$$
\begin{aligned}
&\text{val}      = [1, 2,\ 3,\ 4, 5,\ 6, 7, 8] \\
&\text{col}      = [0, 2,\ 1,\ 0, 1,\ 1, 2, 3] \\
&\text{row\_ptr} = [0,\ 2,\ 3,\ 3,\ 5,\ 8]
\end{aligned}
$$

Computing `y[0]` step by step.

**Step 1:** Find the range of row 0 in `values[]` using `row_ptrs`:

$$
\text{row\_ptr}[0] = 0, \quad \text{row\_ptr}[1] = 2
\implies \text{row 0 spans indices } [0,\ 2)
$$

**Step 2:** Read the values and column indices in that range:

$$
\begin{array}{c|cc}
\text{index} & 0 & 1 \\ \hline
\text{val}   & 1 & 2 \\
\text{col}   & 0 & 2
\end{array}
$$

**Step 3:** Multiply each value by the corresponding element of `x[]`,
using `col_indices[]` to determine *which* element of `x` to use:

$$
y[0] = \text{val}[0] \cdot x[\text{col}[0]] + \text{val}[1] \cdot x[\text{col}[1]]
$$
$$
= 1 \cdot x[0] + 2 \cdot x[2]
$$
$$
= 1 \times 3 + 2 \times 2
$$

> **Note:** Row 2 is all zeros - `row_ptrs[2] = 3, row_ptrs[3] = 3`, so the loop
> runs zero iterations and `y[2] = 0`.

**General formula for any row $i$:**

$$
y[i] = \sum_{j=\text{row\_ptrs}[i]}^{\text{row\_ptrs}[i+1]-1} \text{values}[j] \times x[\text{col\_indices}[j]]
$$
---

## How to Build and Run

```bash
gcc -lm -o run challenge.c
./run
```

---

## Test Results

The test harness runs 100 randomly generated matrices with varying sizes and densities,
checking the output against a reference dense matrix-vector multiplication with a
mixed absolute/relative tolerance of `1e-7`.

| Iterations | Passed | Status |
|---|---|---|
| 100 | 100 | All tests passed |

---

## References

- [CS 357 Illinois - Sparse Matrices (CSR format)](https://cs357.cs.illinois.edu/textbook/notes/sparse.html)
- [Wikipedia - Compressed Sparse Row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
- [CS Emory - Storing Sparse Matrices with Arrays](http://www.cs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse1.html)
