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

We will use the NVIDIA CUDA for Android libraries and include files. However, there are no native Android executables for the CUDA compiler and associated utilities. There are ARM versions for the compiler executable in the Linux for Tegra CUDA toolkit though. The aim here is to get NVIDIA's CUDA compiler nvcc running for PyCUDA under Android. As there are several executables and dependencies on dynamically linked libraries we will use a strategy to temporarily emulate a linux environment using PRoot. A PRoot package is available for Termux and will emulate a linux guest system for the linux executables associated with nvcc temporarily during compilation. For more information on this ingenious piece of magic see:

https://github.com/proot-me/PRoot

Essentially, this is the same idea as used by the GNURoot Debian app but we make much more limited use of the PRoot facilities.

https://play.google.com/store/apps/details?id=com.gnuroot.debian&hl=en

The CUDA executables will need to be linked to Android libraries and running in the Android system. This is necessary as it appears the Android linker will not load and link to libcuda.so from a PRoot environment and the linux dynamic loader appears not capable to link to libcuda.so. The reason for this is not entirely clear and could involve Selinux or setuid restrictions in the later Android versions. Also introduction of a linux version of libcuda.so into the PRoot environment is not succesful as access to the device character files is blocked and libcuda.so will attempt generating the character files through loading the gpu android kernel module via modprobe. Obviously, this requires root access and will fail on a non-rooted device. Installing the Linux for Tegra CUDA toolkit and libcuda.so in the GNURoot Debian PRoot environment does allow the use of the CUDA driver API in Android KitKat 4.4, where the device character files appear accessible from non-root user space.

To install PRoot in Termux you can use the following command at the terminal prompt:

`apt install proot`

The following sections provide detailed instructions on how to get the required files and how to string them together for a working system.

The newest version of NVIDIA CodeWorks for Android 1R5 now provides support for Android Marshmallow and CUDA 7.0 for Tegra X1 devices and is compatible with the higher versions of the Google Pixel C. Notably, CUDA 7.0 is also available for armv7l and will work on the NVIDIA Shield Tablet despite the official support is only for version 6.5. NVIDIA CodeWorks needs to be installed on an Intel/AMD x64 processor bearing PC running Ubuntu 14.04 (AMD64). Point your Web browser to:

https://developer.nvidia.com/gameworksdownload#?dn=codeworks-for-android-1r5

and select **Codeworks for Android 1R5 2016/09/07 DOWNLOADS Ubuntu (64 bit)** in your Web browser. You will be asked to sign into your NVIDIA Developer program account. If not a member yet, it is time to register with NVIDIA and signup, which is usually quick. You can then download the **CodeWorksforAndroid-1R5-linux-x64.run** file. Locate the file in the Downloads folder, add execute permission, and run in an Ubuntu terminal (not on the Pixel C or Shield Tablet):
```
cd Downloads/
chmod +x CodeWorksforAndroid-1R5-linux-x64.run
./CodeWorksforAndroid-1R5-linux-x64.run
```
This will install NVIDIA's Android Studio and the CUDA toolkit. If you are runnuing Ubuntu on an old Laptop that has no CUDA installed, the CUDA toolkit might not install automatically with CodeWorks. It is still possible to get the CUDA toolkit by first installing an older version of AndroidWorks or GameWorks for Android and then upgrading to CodeWorks for Android 1R5 with above run file.

After the installation you should have a full development environment for Android including an Eclipse based IDE setup for native Android development. The files for the Pixel C are in the **~/NVPACK/cuda-android-7.0/aarch64-linux-androideabi/** folder and for the Shield Tablet the files in **~/NVPACK/cuda-android-7.0/armv7-linux-androideabi/** are of interest. The include/ folder contains the CUDA header files, the lib64/ and lib/ folders contains the CUDA libraries including libcudart.so for Android, the samples/ folder with the CUDA samples, and the bin/ folder contains some useful tools for profiling and analysing CUDA applications on Android. Transfer these folders to a portable disk or USB drive.

In Termux navidate to the usr/ folder and add a local/cuda-7.0/ folder:
```
cd ~/../usr
mkdir local
cd local
mkdir cuda-7.0
```
Add a symbolic link to cuda:
```
ln -s cuda-7.0 cuda
```
Copy the files from your portable drive to the Download folder using a file manager App (eg. ES File Explorer). Then copy the files and all subfolders to the usr/local/cuda-7.0/ folder.

**_For the Pixel C:_**
```
cd cuda-7.0
mkdir bin
cp -r ~/storage/shared/Downloads/cuda-android-7.0/aarch64-linux-androideabi/bin/* bin/
mkdir include
cp -r ~/storage/shared/Downloads/cuda-android-7.0/aarch64-linux-androideabi/include/* include/
mkdir lib64
cp -r ~/storage/shared/Downloads/cuda-android-7.0/aarch64-linux-androideabi/lib64/* lib64/
mkdir samples
cp -r ~/storage/shared/Downloads/cuda-android-7.0/aarch64-linux-androideabi/samples/* samples/
```
Now add a symbolic link for lib and libcuda.so on the Pixel C:
```
ln -s lib64/ lib
cd lib64
ln -s /system/vendor64/libnvcompute.so libcuda.so
```

**_For the Shield Tablet:_**
```
cd cuda-7.0
mkdir bin
cp -r ~/storage/shared/Downloads/cuda-android-7.0/armv7-linux-androideabi/bin/* bin/
mkdir include
cp -r ~/storage/shared/Downloads/cuda-android-7.0/armv7-linux-androideabi/include/* include/
mkdir lib64
cp -r ~/storage/shared/Downloads/cuda-android-7.0/armv7-linux-androideabi/lib/* lib/
mkdir samples
cp -r ~/storage/shared/Downloads/cuda-android-7.0/armv7-linux-androideabi/samples/* samples/
```
Add a symbolic link for libcuda.so on the Shield Tablet:
```
cd lib
ln -s /system/vendor/libcuda.so libcuda.so
```

The NVIDIA CUDA for Android files are now in /data/data/com.termux/files/usr/local/cuda-7.0. We need the Linux for Tegra executables for compiling CUDA C++ next.


