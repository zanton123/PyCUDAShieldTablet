# Programming and Running PyCUDA applications on the Google Pixel C (Tegra X1) and NVIDIA Shield (Tegra K1) using CUDA 7.0

This repository contains the files to get PyCUDA running on the Google Pixel C using Termux (by Fredrik Fornwall).

**STATUS:** Pycuda works on the NVIDIA Shield Tablet (original) running Android Marshmallow 6.0.1 and the google Pixel C running Android Nougat 7.0.1 with few limitations. There is a somewhat elaborate installation process for getting nvcc and CUDA 7.0 running. Detail to follow here soon. On the Pixel C MaxAs works both for disassembly and assembly of kernels from SASS, which means can run reassembled kernels from PyCUDA as of today. Unfortunately, MaxAs won't do Keppler code for the shield Tegra K1.


##Install Termux by Fredrik Fornwall from the google play store
https://play.google.com/store/apps/details?id=com.termux

Running Termux opens an Android Terminal with a shell command prompt. This Android shell  features many commands known from Linux distributions. It also comes with a range of packages that have been cross-compiled from Linux to run on native Android, see more at: https://termux.com/

Follow the Termux online help to setup storage and preferences:
https://termux.com/help.html

This tutorial starts with installing several Python packages that are specially useful for seuqence analysis. If you are not a biologist or do not care you can skip through the initial steps, but do install numpy, cython and pycuda obviously.


##Install Python 2.7 in Termux

From the terminal prompt enter the following commands:
```
apt update
apt install clang python2 python2-dev

LDFLAGS=" -lm -lcompiler_rt" pip2 install numpy
LDFLAGS=" -lm -lcompiler_rt" pip2 install pandas
LDFLAGS=" -lcompiler_rt" pip2 install cython
LDFLAGS=" -lcompiler_rt" pip2 install bokeh
LDFLAGS=" -lcompiler_rt" pip2 install HTSeq
```

For installing pysam a linux include file is required that is missing from the Android build system. A dummy wordexp.h file is included in /usr/include/ of this repository or can be generated using the following commands:

`vi /data/data/com.termux/files/usr/include/wordexp.h`

Enter insert mode (`i`) and type:
```
#ifndef _WORDEXP_H_
#define _WORDEXP_H_

#define WRDE_NOCMD 0

typedef struct {
	size_t we_wordc;
	char **we_wordv;
	size_t we_offs;
} wordexp_t;

static inline int wordexp(const char *c, wordexp_t *w, int _i)
{
	return -1;
}

static inline void wordfree(wordexp_t *__wordexp)
{
}

#endif
```

Write and close the file with `Ctrl`+`c` `:wq` ENTER. Then install pysam as follows:

`LDFLAGS=" -lm -lcompiler_rt" pip2 install pysam`

Add pysam library folder /data/data/com.termux/files/usr/lib/python2.7/site-packages/pysam to LD_LIBRARY_PATH (note this is added to the path in the cuda shell script in home/ of this repository):

`export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/data/data/com.termux/files/usr/lib/python2.7/site-packages/pysam`

##NVIDIA CUDA for Android

Nvidia provides support for CUDA development for Android in their CodeWorks for Android toolkit:
https://developer.nvidia.com/codeworks-android

The CUDA toolkit includes several CUDA libraries, of which the CUDA runtime library libcudart.so is most important for running executables with embedded CUDA kernels. Most other libraries provide extended functionality including Fourier transformation, linear algebra and others. Generally, CUDA driver API applications link to libcuda.so, which is supplied by the CUDA driver that is installed with the system and cannot be changed unless the tablet is rooted. Thankfully, the Pixel C comes with a GPU Android kernel module and libnvcompute.so, which is the Pixel C version of the CUDA driver library. For CUDA runntime API applications it appears that libcudart.so dynamically loads libcuda.so (not listed as dynamic dependency in the ELF), which then interfaces with the GPU Android/Linux kernel module. (Linux/Android kernel and CUDA kernels are different). On the Pixel C libnvcompute.so can be used instead of libcuda.so. The NVIDIA Shield Tablet comes with a regular libcuda.so.

We will use the NVIDIA CUDA for Android libraries and include files. However, there are no native Android executables for the CUDA compiler and associated utilities. The Linux for Tegra CUDA toolkit for the compiler executables. The aim here is to habe NVIDIA's CUDA compiler nvcc running for PyCUDA. As there are several executables and dynamically linked libraries we will use a PRroot strategy. A PRoot package is available for Termux and will emulate a linux guest system for nvcc temporarily during compilation. For more information on this ingenious piece of magic see:

https://github.com/proot-me/PRoot

Essentially, the idea is the same as of the GNURoot Debian app but we make much more limited use of the PRoot facilities.

https://play.google.com/store/apps/details?id=com.gnuroot.debian&hl=en

The CUDA executables will be linked to Android libraries and running in the Android system. This is necessary as it appears the Android linker will not load and link to libcuda.so from a PRoot environment and the linux dynamic loader appears not capable to link to libcuda.so. The reason for this is not entirely clear and could involve Selinux or setuid restrictions in the later Android versions. Installing the Linux for Tegra CUDA toolkit and using libcuda.so in GNURoot Debian allow the use of the CUDA driver API in Android KitKat 4.4.


The following sections provide detailed instructions on how to get the required files and how to string them together for a working system.


