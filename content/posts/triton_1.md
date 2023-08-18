---
title: "Triton Tutorial 1: Vector Addition"
date: 2023-08-01
draft: false
---

In this exercise we will create a very simple vector addition kernel in Triton. Vector addition in PyTorch is as simple as 
```python
import torch

x = randn(7)
y = randn(7)
out = x + y  # vector addition
```
This exercise is intended to familarize yourself with the basics of Triton and the Triton language.
## The Vector Addition Kernel
If you have read the introduction[^1], you now know that Triton implements *blocked algorithms*. If we have two vectors of size 7 and a `BLOCK_SIZE` of 3, we will end up with `triton.cdiv(7, 3)` = 3 parallel programs, where each program works on a chunk of the data:
![Two sequences "packed".](/img/triton/triton_lang_blocked.svg)

In the Triton kernel we can check what chunk we are working on with the program ID variable `triton_language.program_id(axis=0)`[^3], which will evaluate <span style="background-color:#f4ccccff">pid=0</span>, <span style="background-color:#d9ead3ff">pid=1</span> and <span style="background-color:#d9d2e9ff">pid=2</span>. This is pretty straightforward, if you have experience with CUDA it will feel similar to `idx = blockIdx.x + threadIdx.x`.

Where it does differ greatly from CUDA is that we can load and store data using a tensor/list of pointers. For example, to load the block of data required for <span style="background-color:#f4ccccff">pid=0</span>, we create a range of values from 0 to `BLOCK_SIZE` and add the starting index of the block (`pid * BLOCK_SIZE`) to it, resulting in <span style="background-color:#f4ccccff">offsets=[0, 1, 2]</span>. Retrieving the block values from an input vector is then as simple as adding its pointer to the offsets: `triton_language.load(ptr_to_first_vector + offsets)`. This will load the first block completely into super fast SRAM, allowing the hardware to perform fast math operations before storing it again back to DRAM using `triton_language.store(ptr_to_output_vector + offsets)`. The docs for both [load](https://triton-lang.org/main/python-api/generated/triton.language.load.html#triton-language-load) and [store](https://triton-lang.org/main/python-api/generated/triton.language.store.html#triton-language-store) are considered required reading, there is a diverse set of input options here.

You might figure out that the offsets for <span style="background-color:#d9d2e9ff">pid=2</span> will not make much sense, it will evaluate to <span style="background-color:#d9d2e9ff">offsets=[6, 7, 8]</span> - index 7 and 8 are out of bounds for a vector of size 7. To avoid accessing out-of-bound memory, both the load and the store functions have a `mask` argument. We can pass a bool/int1 tensor here that should be the same length as `BLOCK_SIZE` with 1's indicating valid memory. Triton will simply not load/store anything where the mask is 0'd out.
## Launching the Kernel
With the kernel loading the data to SRAM, adding the blocks together, and storing it back to DRAM finished, we can now look into how we actually call the kernel. First, the kernel needs a specific just-in-time decorator: `triton.jit`. This decorator does a lot of work, lets read the [docs](https://triton-lang.org/main/python-api/generated/triton.jit.html#triton.jit):

> Note: When a jit’d function is called, arguments are implicitly ***converted to pointers*** if they have a `.data_ptr()` method and a .dtype attribute. 
> Note: This function will be compiled and run on the GPU. It will ***only have access to***:
> - python primitives,
> - builtins within the triton package,
> - arguments to this function,
> - other jit’d functions

Good to know that any tensor will be converted to a pointer *to its first element*, and that the kernel itself can only be written using Triton and very basic Python primitives! So how do we go about launching it? Well, another thing that `triton.jit` does is adding an input argument that defiens the launch *grid*. The grid, in it simplest form, defines how many kernel instances we want to run in parallel. A small snippet:
```python
import triton

@triton.jit
def f_kernel(x_ptr, y_ptr, out_ptr):
	# load x, y, do ops, store out, return nothing

def f(x, y):
	out = torch.empty_like(x)

	# run the kernel, indicating the launch grid.
	# Note: it's a tuple for each axis.
	launch_grid = (triton.cdiv(7, 3), )
	f_kernel[launch_grid](x, y, out)

	return out
```
The syntax felt a bit funky to me, but Numba does basically [the same](https://numba.pydata.org/numba-doc/dev/cuda/kernels.html). We will dive deeper into the launch grid in subsequent exercises, it turns out that we can define the grid in many different ways.
## Exercise
Implement the vector addition kernel and the launch function in [vector_addition_kernel.py](https://github.com/lweitkamp/triton_tutorial/blob/main/vector_addition/vector_addition_kernel.py). If you get stuck, solutions can also be found in the [solutions branch](https://github.com/lweitkamp/triton_tutorial/tree/solutions) - but for this exercise you should not need it. After you are done coding, run the test case to ensure that the results of the Triton kernel are equal to that of a native PyTorch vector addition. You can run [vector_addition_benchmark.py](https://github.com/lweitkamp/triton_tutorial/blob/main/vector_addition/vector_addition_benchmark.py) and it should look a bit as follows:

...

These results make sense. Vector addition is not a difficult problem and cannot really be optimized further, it is inherently memory-bound.

[^1]: link to introduction
[^3]: the [documentation](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html#triton.language.program_id). Here we set `axis=0` - we are using a 1D grid since our inputs/outputs are vectors.
