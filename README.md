Nvidia Tutorial: An Even Easier Introduction to CUDA
===

[Tutorial Link](https://colab.research.google.com/github/NVDLI/notebooks/blob/master/even-easier-cuda/An_Even_Easier_Introduction_to_CUDA.ipynb#scrollTo=BEijwk25id3t)

## Purpose of the Leacture

- Launch massively parallel CUDA Kernels on an NVIDIA GPU
- Organize parallel thread execution for massive dataset sizes
- Manage memory between the CPU and GPU
- Profile your CUDA code to observe performance gains

## Starting Simple

```cpp=
%%writefile add.cpp

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}
```

```bash=
g++ add.cpp -o add
./add
```

### CUDA Kernel function

- Turn our `add` function into a function that the GPU can run, called a kernel in CUDA. To do this, all I have to do is add the specifier `__global__` to the function, which tells the CUDA C++ compiler that this is a function that **runs on the GPU** and can be called from CPU code.

```cpp=
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
```

- These `__global__` functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.

### Memory Allocation in CUDA

- To allocate data in unified memory, call `cudaMallocManaged()`, which returns a **pointer** that you can access from host (CPU) code or device (GPU) code.

```cpp=
// Allocate Unified Memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
```

- To free the data, just pass the pointer to `cudaFree()`

```cpp=
  // Free memory
  cudaFree(x);
  cudaFree(y);
```

- To launch the `add()` kernel, which invokes it on the GPU. CUDA kernel launches are specified using the triple angle bracket syntax `<<< >>>`.

```cpp=
  // this line launches one GPU thread to run add()
  add<<<1, 1>>>(N, x, y);
```

- Just one more thing: force the CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). 
- To do this just call `cudaDeviceSynchronize()` before doing the final error checking on the CPU.

---

Here’s the complete CUDA code:

```cpp=
%%writefile add.cu

#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

```bash=
nvcc add.cu -o add_cuda
./add_cuda
```

- This kernel is only correct for a single thread, since every thread that runs it will perform the add on the whole array.
- Moreover, there is a **race condition** since multiple parallel threads would both read and write the same locations.

## Profile it

- The simplest way to find out how long the kernel takes to run is to run it with `nvprof`, the command line GPU profiler that comes with the CUDA Toolkit. Just type `nvprof ./add_cuda` on the command line:

```bash=
nvprof ./add_cuda
```

- To see the current GPU allocated to you run the following cell and look in the Name column where you might see.

```bash=
nvidia-smi
```

## Picking up the Threads

- Now that you’ve run a kernel with one thread that does some computation, how do you make it parallel? The key is in CUDA’s `<<<1, 1>>>` syntax. 
- This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU. 
- There are two parameters here, but let’s start by changing **the second one:** the number of threads in a thread block. CUDA GPUs run kernels using blocks of threads that are a **multiple of 32 in size**, so **256 threads is a reasonable** size to choose.

```cpp=
add<<<1, 256>>>(N, x, y);
```

- If run the code with only this change, it will do the computation once per thread, rather than spreading the computation across the parallel threads.
- To do it properly, I need to **modify the kernel**. CUDA C++ provides keywords that let kernels get the indices of the running threads.
- Specifically, `threadIdx.x` contains the index of the current thread within its block, and `blockDim.x` contains the number of threads in the block. **Just modify the loop to stride through the array with parallel threads.**

```cpp=
__global__ 
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
```

- The `add` function hasn’t changed that much. In fact, setting index to 0 and stride to 1 makes it semantically identical to the first version.
- Here we save the file as `add_block.cu` and compile and run it in `nvprof` again.

```cpp=
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

```bash=
nvcc add_block.cu -o add_block
nvprof ./add_block
```

- That’s a big speedup (compare the time for the add kernel by looking at the GPU activities field), but not surprising since I went from 1 thread to 256 threads. Let’s keep going to get even more performance

### Out of the Blocks

- By now you may have guessed that **the first parameter** of the execution configuration specifies the number of thread blocks.
- Together, the blocks of parallel threads make up what is known as the grid. Since I have `N` elements to process, and 256 threads per block, I just need to calculate the number of blocks to get at least `N` threads.
- **I simply divide `N` by the block size** (being careful to round up **in case `N` is not a multiple of blockSize**).

```cpp=
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```

![](https://hackmd.io/_uploads/S1G1sjohn.png)

- I also need to update the kernel code to take into account the entire grid of thread blocks. 
- `gridDim.x`, which contains the number of blocks in the grid.
- `blockIdx.x`, which contains the index of the current thread block in the grid. 
- Figure 1 illustrates the the approach to indexing into an array (one-dimensional) in CUDA using `blockDim.x`, `gridDim.`x, and `threadIdx.x`. 
- The idea is that each thread gets its index by **computing the offset** to the beginning of its block (the block index times the block size: `blockIdx.x * blockDim.x`) and adding the thread’s index within the block (`threadIdx.x`). The code **`blockIdx.x * blockDim.x + threadIdx.x`** is idiomatic CUDA.

```cpp=
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```

- The updated kernel also sets stride to the total number of threads in the grid `(blockDim.x * gridDim.x)`. This type of loop in a CUDA kernel is often called a grid-stride loop.

---

- Save the file as `add_grid.cu` and compile and run it in `nvprof` again.

```cpp=
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

```bash=
nvcc add_grid.cu -o add_grid
nvprof ./add_grid
```

## Exercise

To keep you going, here are a few things to try on your own.

1. Experiment with `printf()` inside the kernel. Try printing out the values of `threadIdx.x` and `blockIdx.x` for some or all of the threads. Do they print in sequential order? Why or why not?
2. Print the value of `threadIdx.y` or `threadIdx.z` (or `blockIdx.y`) in the kernel. (Likewise for `blockDim` and `gridDim`). Why do these exist? How do you get them to take on values other than 0 (1 for the dims)?