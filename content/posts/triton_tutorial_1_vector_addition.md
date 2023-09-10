---
title: "Triton Tutorial 1: Vector Addition"
date: 2023-08-02
draft: false
---

This is part 1 of a tutorial series on Triton. Find the other parts here:
- [Part 0: Introduction]({{< ref "triton_tutorial_0_introduction" >}})
- [Part 1: Vector Addition]({{< ref "triton_tutorial_1_vector_addition" >}})
- [Part 2: Softmax]({{< ref "triton_tutorial_2_softmax" >}})
- [Part 3: Blocked Sum Reduction]({{< ref "triton_tutorial_3_blocked_sum_reduction" >}})
- [Part 4: Matrix Multiplication]({{< ref "triton_tutorial_4_matrix_multiplication" >}})

# Introduction
In this assignment we will create a very simple vector addition kernel in Triton.
Vector addition in PyTorch is as simple as:

```python
import torch

x = randn(7)
y = randn(7)
out = x + y  # vector addition
```
This assignment is intended to familarize yourself with the basics of Triton and the Triton language.
For some terminology, please refer to the [introduction](https://lweitkamp.github.io/posts/triton_tutorial_0_introduction/).
The following topics are covered in this assignment:
- Figuring out what block of data the kernel will work on
- Loading data from DRAM to SRAM, saving from SRAM to DRAM
- A small introduction to the *launch grid*

# The Vector Addition Kernel
For a vector addition kernel, the inputs will hence be three vectors: `x`, `y`, and `out`.
The kernel  will load the datafrom `x` and `y`, add them together, and store the result in `out`.
If you have read the introduction[^1], you now know that Triton implements *blocked algorithms*. If we have two vectors of size 7 and a `BLOCK_SIZE` of 3, we will end up with `triton.cdiv(7, 3)` = 3 parallel programs, where each program works on a chunk of the data:


{{< figure src="/img/triton/triton_lang_blocked.svg" caption="An example of blocks of size three on two vectors, both of size 7. The first vector spells 'T R I T O N L', the second spells 'A N G U A G E'." >}}

## JIT
A function can be turned into a triton kernel simply by decorating it with [`triton.jit`](https://triton-lang.org/main/python-api/generated/triton.jit.html#triton.jit). The following code snippet is an example:

```python
import triton
import torch

@triton.jit
def f_kernel(x_ptr, y_ptr, out_ptr):
    pass

out = torch.empty(7)
f_kernel(torch.randn(7), torch.randn(7), out)
```
We decorate the `f_kernel` function with `triton.jit`, which will let Triton just-in-time compile it and will subsequently call it with three tensors. The input arguments are implicitly converted to pointers to their first element if they are tensors. the JIT decorator adds a lot of variables here:

| Parameter | Explanation |
| :-------- | :---------- |
| `num_warps` | the number of warps to use for the kernel when compiled for GPUs. For example, if num_warps=8, then each kernel instance will be automatically parallelized to cooperatively execute using 8 * 32 = 256 threads. [Here](https://softwareengineering.stackexchange.com/a/369978) is a nice stackoverflow explainer. [from Triton Config](https://triton-lang.org/main/python-api/generated/triton.Config.html). |
| `num_ctas` | An integer denoting the number of *Cooperative Thread Arrays* active. Think of a CTA as a set of warps for now. In NVIDIA terms it is a block, but that term is a bit confusing in a blocked algorithm discussion. |
| `num_stages` |  the number of stages that the compiler should use when software-pipelining loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs (A100s).  [from Triton Config](https://triton-lang.org/main/python-api/generated/triton.Config.html) and [an explainer](https://github.com/openai/triton/discussions/512#discussioncomment-2732003) by the author of Triton. |
| `enable_warp_specialization` | Enables [warp specialization](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization). |
| `extern_libs` | An optional dictionary with name of library as key and path to library as value. Triton jitted kernels are limited in operations they can perform. We can load in externals libraries that have math functions not included in the basic Triton package. [From Triton docs](https://triton-lang.org/main/getting-started/tutorials/07-math-functions.html). | 
| `stream` | The [PyTorch CUDA stream explainer](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams) might be helpful. | 
| `device` | If none, will fetch the current device `torch.cuda.current_device()`, otherwise expects an integer index to set as device using `torch.cuda.set_device(idx)` |
| `device_type` | A string denoting the device type (typically 'cuda') |

Note that at the time of writing documentation is a bit sparse on these parameters. If you want to dive deeper, look at this [code block](https://github.com/openai/triton/blob/a7b40a10f98a748256c69cdc9efc742fc9617899/python/triton/runtime/jit.py#L365C1-L365C1). We won't need most of these parameters for this tutorial series, but it's good to know they exist.

## Program Identifier
When a kernel launches, several programs will be run in parallel. We can check what block of data this specific program is working on with the program ID variable [`triton_language.program_id(axis=0)`](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html#triton.language.program_id), which will evaluate <span style="background-color:#f4ccccff">pid=0</span>, <span style="background-color:#d9ead3ff">pid=1</span> and <span style="background-color:#d9d2e9ff">pid=2</span>.
We use the program ID variable to create offsets in the tensors we work on. It is similar to how we would do this in CUDA: `idx = blockIdx.x + threadIdx.x`, but the big difference is that the offsets are lists/tensors in Triton, **not scalars**.
Since we are working on a simple vector multiplication operation, the offsets are only dependent on the program ID and the chosen `BLOCK_SIZE` constant:

| <div style="width:100px">Program ID</div> | <div style="width:200px">Offsets</div> | <div style="width:200px">Vector 1 Block</div> | <div style="width:200px">Vector 2 Block</div> |
| :-- | :------ | :------------ | :------------ |
| <span style="background-color:#f4ccccff">pid=0</span> | <span style="background-color:#f4ccccff">offsets=[0, 1, 2]</span> | <span style="background-color:#f4ccccff">t r i</span> | <span style="background-color:#f4ccccff">a n g</span>  |
| <span style="background-color:#d9ead3ff">pid=1</span> | <span style="background-color:#d9ead3ff">offsets=[3, 4, 5]</span> | <span style="background-color:#d9ead3ff">t o n</span> | <span style="background-color:#d9ead3ff">u a g</span>  |
| <span style="background-color:#d9d2e9ff">pid=2</span> | <span style="background-color:#d9d2e9ff">offsets=[6, 7, 8]</span> | <span style="background-color:#d9d2e9ff">l</span> | <span style="background-color:#d9d2e9ff">e</span>  |

Creating offsets is easy. If you have the program ID ready and the `BLOCK_SIZE` value, you can use [`triton_language.arange`](https://triton-lang.org/main/python-api/generated/triton.language.arange.html) to create indices from the start of the block all the way to the end.

## Loading Data to SRAM
Now we know what block we will be working on, it's time to get this data from DRAM to SRAM.
Retrieving the block values from an input vector is very easy in Triton: [`triton_language.load(ptr_to_first_vector + offsets)`](https://triton-lang.org/main/python-api/generated/triton.language.load.html), thats all!
However, you might have noticed that for <span style="background-color:#d9d2e9ff">pid=2</span> the offsets will overshoot the data and might access out-of-bound memory.
To ensure we do not run into any memory errors, we can use a `mask` argument in the `load` function. In our case, the mask is a bool tensor with the same length as `BLOCK_SIZE`, where 1's indicate valid memory. Triton will simply not load anything where the mask is 0'd out.

## Addition and Storing Data Back to DRAM
Tensors are first-class citizens in Triton, which makes this part very simply, not worthy of a code block: `out_block = x_block + y_block`.
With that done, all that is left is storing `out_block` in its proper location using the [`triton_language.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html) function (which also accepts a `mask` argument).

That's all we have to do for the kernel. In summary, we use the program ID and the `BLOCK_SIZE` constant to figure out what block this specific kernel has to work on by calculating the offsets. The offsets can be used with pointers to the input and output tensors to load and store data, respectively. As you saw, tensors are first-class citizens in Triton, so we could've just as easily multiplied/divided/exponentiated them together instead of adding them. We will cover some of these in next assignments.


# Launching the Kernel
The kernel is ready to launch, so lets discuss how to approach that. First, we need to create a function that will launch the kernel. This function will be called from Python, and will be responsible for allocating the memory for the input and output tensors, and for calling the kernel:

```python
import triton
import torch

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

y = f(torch.randn(7), torch.randn(7))
```

## The Launch Grid
From the snippet above, only the *launch grid* should look new.
This is the result of `triton.jit`; it will add a dictionary-style input argument to the kernel function that defines the number of kernels we run in parallel.
In it's simplest form, the launch grid is a tuple of integers, where each integer defines the number of kernel instances to launch in parallel for each axis.
Since we have a 1D problem, the launch grid is a tuple with a single integer value.


If we have a defined `BLOCK_SIZE`, it makes a lot of sense to split the input vectors into blocks of size `BLOCK_SIZE` and run a kernel on each block.
If there are not enough cores to run all the blocks in parallel, Triton will automatically schedule the blocks in a way that all cores are used as much as possible.
In the example above, the vectors are of size 7 and `BLOCK_SIZE` is 3, so we will run 3 kernels in parallel, where each kernel processes 3 elements.


The syntax felt a bit funky to me, but Numba does basically [the same](https://numba.pydata.org/numba-doc/dev/cuda/kernels.html). We will dive deeper into the launch grid in subsequent assignments, it turns out that we can define the grid in many different ways.

# The Assignment
Implement the vector addition kernel and the launch function in [vector_addition_kernel.py](https://github.com/lweitkamp/triton_tutorial/blob/main/vector_addition/vector_addition_kernel.py). If you get stuck, solutions can also be found in the [solutions branch](https://github.com/lweitkamp/triton_tutorial/tree/solutions) - but for this assignment you should not need it. After you are done coding, run [vector_addition_test.py](https://github.com/lweitkamp/triton_tutorial/blob/main/vector_addition/vector_addition_test.py) to ensure that the results of the Triton kernel are equal to that of a native PyTorch vector addition. You can run [vector_addition_benchmark.py](https://github.com/lweitkamp/triton_tutorial/blob/main/vector_addition/vector_addition_benchmark.py) and it should look a bit as follows:

![Benchmark results of the vector addition kernel. The figure depicts two curves very close to eachother.](/img/triton/vector-add-performance.png)

It's very easy to get peak cuBLAS-like performance for vector addition, since vector adition is bandwidth-bound[^1]: vector addition for a vector of size N takes N arithmetic operations, but 3N memory operations (2 reads, 1 write). For now, don't worry about the benchmarking code, we will cover that in a future assignment.


[^1]: [This post by Robert Crovella](https://forums.developer.nvidia.com/t/sequential-code-is-faster-than-parallel-how-is-it-possible/44165/2) explains it in more detail. He also teaches the basics of CUDA in the [Oak Ridge CUDA training series](https://www.olcf.ornl.gov/cuda-training-series/), a recommend if you want to get started on CUDA but also good to get a beter understanding of NVIDIA GPUs.