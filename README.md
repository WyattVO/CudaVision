
# CUDA & C++ Performance Lab
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
> An Aggie Coding Club project for learning C++, the CUDA stack, and how to make code *fast*.
<img width="1805" height="1024" alt="image" src="https://github.com/user-attachments/assets/15f5e843-39e8-44c3-b2ca-52e54f930e34" />

## About

The CUDA & C++ Performance Lab is a hands-on, learning-focused project that takes students from the fundamentals of C++ to writing GPU-accelerated code with NVIDIA's CUDA stack.

It's open to **Freshmen through Seniors at any coding level**. No prior C++ or GPU experience is required, just curiosity.

Each exercise pairs a CPU implementation with a CUDA version, so you can measure the speedup yourself and understand *why* it happened.

## What You'll Learn

- **C++ fundamentals:** syntax, types, pointers, memory, and building/compiling programs
- **General programming tips:** clean code, debugging, version control, and good habits
- **Parallel thinking:** how to break problems into pieces that run at the same time
- **CUDA basics:** kernels, threads, blocks, and grids
- **GPU memory:** host vs. device memory, transfers, and shared memory
- **Performance & profiling:** timing code, finding bottlenecks, and optimizing

## How It Works

1. **Learn the concept:** a short explanation of the topic for that exercise
2. **Write the CPU version:** solve the problem in plain C++
3. **Port it to CUDA:** move the work onto the GPU
4. **Measure it:** benchmark both versions and compare
5. **Optimize:** apply what you learned to squeeze out more performance

## Repository Structure

```
.
├── 00-setup/            # Environment setup and first program
├── 01-cpp-basics/       # Core C++ exercises
├── 02-parallel-intro/   # Intro to parallel thinking
├── 03-cuda-basics/      # First CUDA kernels
├── 04-memory/           # GPU memory management
├── 05-optimization/     # Profiling and performance tuning
└── README.md
```

> Structure may change as new exercises are added.

## Getting Started

### Prerequisites

- A C++ compiler (`g++` or `clang++`)
- An NVIDIA GPU with CUDA support (or access to a lab/cloud machine with one)
- The [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Git

### Setup

```bash
git clone <repo-url>
cd cuda-cpp-performance-lab
```

Check that CUDA is installed:

```bash
nvcc --version
nvidia-smi
```

### Compile and Run an Exercise

```bash
# C++ version
g++ -O2 main.cpp -o main
./main

# CUDA version
nvcc -O2 main.cu -o main_cuda
./main_cuda
```

> No NVIDIA GPU? Reach out to the project leads about lab machines or free cloud options like Google Colab.

## Who This Is For

- Freshmen just starting to code
- Students who know another language and want to learn C++
- Anyone curious about GPUs, parallel computing, or performance
- Upperclassmen looking to build systems and HPC skills

## Contributing

New exercises, fixes, and explanations are welcome.

1. Fork the repo
2. Create a branch (`git checkout -b add-new-exercise`)
3. Commit your changes
4. Open a pull request

## Contact

Questions? Find us at Aggie Coding Club meetings or reach out to the project leads.

---

*Built by students, for students. Gig 'em!*
