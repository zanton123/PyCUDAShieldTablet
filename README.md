# Programming and Running PyCUDA applications on the Google Pixel C (Tegra X1) using CUDA 7.0

This repository contains the files to get PyCUDA running on the Google Pixel C using Termux (by Fredrik Fornwall).

**STATUS:** Pycuda works on the NVIDIA Shield Tablet (original) running Android Marshmallow 6.0.1 and the google Pixel C running Android Nougat 7.0.1 with few limitations. There is a somewhat elaborate installation process for getting nvcc and CUDA 7.0 running. Detail to follow here soon. On the Pixel C MaxAs works both for disassembly and assembly of kernels from SASS, which means can run reassembled kernels from PyCUDA as of today. Unfortunately, MaxAs won't do Keppler code for the shield Tegra K1.
