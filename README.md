# Programming and Running PyCUDA applications on the Google Pixel C (Tegra X1) and NVIDIA Shield (Tegra K1) using CUDA 7.0

This repository contains the files to get PyCUDA running on the Google Pixel C using Termux (by Fredrik Fornwall).

**STATUS:** Pycuda works on the NVIDIA Shield Tablet (original) running Android Marshmallow 6.0.1 and the google Pixel C running Android Nougat 7.0.1 with few limitations. There is a somewhat elaborate installation process for getting nvcc and CUDA 7.0 running. Detail to follow here soon. On the Pixel C MaxAs works both for disassembly and assembly of kernels from SASS, which means can run reassembled kernels from PyCUDA as of today. Unfortunately, MaxAs won't do Keppler code for the shield Tegra K1.

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

Write and close the file with `Ctrl`+`c` `:wq` ENTER. The install pysam as follows:

`LDFLAGS=" -lm -lcompiler_rt" pip2 install pysam`

Add pysam library folder /data/data/com.termux/files/usr/lib/python2.7/site-packages/pysam to LD_LIBRARY_PATH (note this is added to the path in the cuda shell script in home/ of this repository):

`export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/data/data/com.termux/files/usr/lib/python2.7/site-packages/pysam`


