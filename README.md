# CudaScripts
Experiments with CUDA and GPU programming.

The code is developed and compiled on the machine `ray17`.

### Prerequisites
Make sure you have the following installed:
- A **CUDA-capable NVIDIA graphics card**.
- **NVIDIA drivers**, **CUDA**, and a **suitable version of GCC**.

### Switching between GCC versions
How can we switch between different GCC versions on the lab machines and GPU clusters?

- On `ray17`, the CUDA driver and compatible GCC version should be loaded correctly.
- **CUDA version and GCC compatibility**: When logging into the lab machine, ensure you load the CUDA driver.
- You can check available CUDA versions by running:
    ```bash
    ls /vol/cuda/
    ```
- On ray17, the following command loads **CUDA v12.5.0**:
    ```bash
    . /vol/cuda/12.5.0/setup.sh
    ```
- This version is compatible with **GCC v13.3.0**.
- **Be cautious**: Initially loading **CUDA v12.0** and using the pre-installed GCC resulted in compilation errors. Therefore, make sure the version of **GCC** is compatible with the **NVIDIA CUDA Compiler** (`nvcc`).

After loading the correct environment, compile with:
  ```bash
  nvcc gpu_info.cu -o gpu_info
  ```

This should compile without warnings.

### Required ingredients for compatibility
Ensure that the following tools and versions are compatible:
- GCC, nvcc, and nvidia-smi should harmonise for successful compilation.
- Example of compatible versions:
  - GCC version:
    ```bash
    USERNAME@ray17:CUDAScripts$ gcc --version
    gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
    Copyright (C) 2023 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    ```

  - CUDA version
    ```bash
    USERNAME@ray17:CUDAScripts$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Wed_Apr_17_19:19:55_PDT_2024
    Cuda compilation tools, release 12.5, V12.5.40
    Build cuda_12.5.r12.5/compiler.34177558_0
    ```

  - NVIDIA driver and GPU information:
    ```bash
    USERNAME@ray17:CUDAScripts$ nvidia-smi
    Mon Mar 31 20:35:47 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce GTX 1080        Off |   00000000:01:00.0 Off |                  N/A |
    | 29%   28C    P8              6W /  180W |      83MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                     
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |    0   N/A  N/A   1066597      G   /usr/lib/xorg/Xorg                             77MiB |
    +-----------------------------------------------------------------------------------------+
    ```

### Summary
- Ensure that GCC, nvcc, and nvidia-smi are compatible for seamless compilation.
- Always double-check the compatibility between `CUDA`, `gcc`, and your GPU drivers for optimal performance.
- NOTE: For high-level applications like `TensorFlow`, additional dependencies might be required, and their versions must also be compatible with the installed version of `CUDA`.
