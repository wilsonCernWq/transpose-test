# CUDA
Outputs:
```
user@gpu:/workspace$ ./transpose 

Device : NVIDIA GeForce RTX 4090
Matrix size: 1024 1024, Block size: 32 8, Tile size: 32 32
dimGrid: 32 32 1. dimBlock: 32 8 1
                  Routine         Bandwidth (GB/s)
                     copy             2293.67
       shared memory copy             2281.89
          naive transpose              271.71
      coalesced transpose             1383.78
  conflict-free transpose             2267.09
```


# SYCL (Fail due to Incorrect)
Outputs:
```
root@c993425f9cbb:/workspace# icpx -fsycl ./dpct_output/transpose.dp.cpp 
root@c993425f9cbb:/workspace# ./a.out 

Device : Intel(R) Arc(TM) A770 Graphics
Matrix size: 1024 1024, Block size: 32 8, Tile size: 32 32
dimGrid: 32 32 1. dimBlock: 32 8 1
max_work_group_size: 1024
has_local_mem: 1
global_mem_size: 16.225243 GB
local_mem_size: 65536 Byte
                  Routine         Bandwidth (GB/s)
                     copy              122.59
       shared memory copy
#1 0.000000 1.000000
           *** FAILED ***
          naive transpose              149.69
      coalesced transpose              271.19
  conflict-free transpose
#2 0.000000 2048.000000
           *** FAILED ***
```

Environment:
```
root@c993425f9cbb:/workspace# icpx --version
Intel(R) oneAPI DPC++/C++ Compiler 2024.0.2 (2024.0.2.20231213)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/intel/oneapi/compiler/2024.0/bin/compiler
Configuration file: /opt/intel/oneapi/compiler/2024.0/bin/compiler/../icpx.cfg
```
```
root@c993425f9cbb:/workspace# c2s --version
dpct version 18.0.0. Codebase:(a183f90429f8fb792ae1f2fe70bd5b86dba6deed)
```
