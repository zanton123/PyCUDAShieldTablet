# Programming and Running PyCUDA applications on the Google Pixel C (Tegra X1) and NVIDIA Shield Tablet (Tegra K1) using CUDA 7.0

This repository contains the files to get PyCUDA running on the Google Pixel C using Termux (by Fredrik Fornwall).

**STATUS:** Pycuda works on the NVIDIA Shield Tablet (original) running Android Marshmallow 6.0.1 and the google Pixel C running Android Nougat 7.0.1 with few limitations. There is a somewhat elaborate installation process for getting nvcc and CUDA 7.0 running. Detail to follow down here. On the Pixel C MaxAs works both for disassembly and assembly of kernels from SASS, which means can run reassembled kernels from PyCUDA. Unfortunately, MaxAs won't do Keppler code for the Shield Tegra K1.


##Install Termux by Fredrik Fornwall from the google play store
https://play.google.com/store/apps/details?id=com.termux

Running Termux opens an Android Terminal with a shell command prompt. This Android shell  features many commands known from Linux distributions. It also comes with a range of packages that have been cross-compiled from Linux to run on native Android, see more at: https://termux.com/

Follow the Termux online help to setup storage and preferences:
https://termux.com/help.html


##Install Python 2.7 in Termux

This tutorial starts with installing several useful packages in Termux. Python packages that are specially useful for sequence analysis might not be of interest to people that are not biologists or do not care. You can by all means skip through these, but do install numpy, cython and pycuda obviously these are needed.

From the terminal prompt enter the following commands:
```
apt update
apt install clang python2 python2-dev
apt install build-essential, cmake, make, perl, wget

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

The NVIDIA CUDA for Android files are now in /data/data/com.termux/files/usr/local/cuda-7.0. We will install the Linux for Tegra executables for compiling CUDA C++ next.

##NVIDIA Linux for Tegra

NVIDIA also does not supply CUDA compiler components that run natively on Android (the toolkit contains only profiling and analysis tools ported to Android). However, both armhf for the Tegra K1 and ARM64 for the Tegra X1 versions are in the Linux for Tegra Archive: 

https://developer.nvidia.com/embedded/linux-tegra-archive

Jetson TX1 R24.1 – May 2016 contains the CUDA 7.0 tools matching our CUDA installation and can be downloaded from:

https://developer.nvidia.com/embedded/downloads

Scroll down the list and select **JetPack for L4T 2.2 2016/06/15** for download on your Ubuntu PC. Locate **JetPack-L4T-2.2-linux-x64.run** file in an Ubuntu terminal and run:
```
cd ~/Downloads
chmod +x JetPack-L4T-2.2-linux-x64.run
./JetPack-L4T-2.2-linux-x64.run
```
When prompted select first JETPACK 2.2 – Tegra X1 64 bit and then rerun to select JETPACK 2.2 - Tegra X1 32 bit. The installer will download all files and then ask to flash the operating system image to a Jetson X1 development board. At this point the installation can be aborted. In the **~/Downloads/Jetpack_2.2/jetpack_download/** folder are two archives: **cuda-repo-l4t-7-0-local_7.0-76_arm64.deb** contains the ARM64 CUDA 7.0 toolkit and **Tegra_Linux_Sample-Root-Filesystem_R24.1.0_aarch64.tbz2** contains the ARM64 Linux root file system for the Pixel C. The respective armhf archives for the Shield Tablet are **cuda-repo-l4t-7-0-local_7.0-76_armhf.deb** and **Tegra_Linux_Sample-Root-Filesystem_R24.1.0_armhf.tbz2**. Use Ubuntu Archive Manager to open these archives and then drag and drop between file manager windows to extract and copy files and folders from these archieves (on the file right click the mouse to see the options for open).

From the **Tegra_Linux_Sample-Root-Filesystem_R24.1.0_aarch64.tbz2** archive copy the following files from the **/lib/aarch64-linux-gnu/** to a portable drive. Then copy the files/ folder to the Download folder on the Pixel C and move them to a new folder structure under the /data/data/com.termux/files/ termux main folder The file structure should be as follows:
```
/data/data/com.termux/files/
├── bin
│   ├── bunzip2
│   ├── bzip2
│   ├── cat
│   ├── chmod
│   ├── cp
│   ├── dash
│   ├── date
│   ├── echo
│   ├── egrep
│   ├── grep
│   ├── gunzip
│   ├── gzip
│   ├── mkdir
│   ├── mv
│   ├── rm
│   ├── rmdir
│   ├── sed
│   ├── sh -> dash
│   ├── tar
│   └── which
│
├── home
│   ├── cuda
│   └── normal
│
├── lib
│   ├── aarch64-linux-gnu
│   │   ├── ld-2.19.so
│   │   ├── ld-linux-aarch64.so.1 -> ld-2.19.so
│   │   ├── libacl.so.1 -> libacl.so.1.1.0
│   │   ├── libacl.so.1.1.0
│   │   ├── libattr.so.1 -> libattr.so.1.1.0
│   │   ├── libattr.so.1.1.0
│   │   ├── libc-2.19.so
│   │   ├── libc.so.6 -> libc-2.19.so
│   │   ├── libdl-2.19.so
│   │   ├── libdl.so.2 -> libdl-2.19.so
│   │   ├── libgcc_s.so.1
│   │   ├── libm-2.19.so
│   │   ├── libm.so.6 -> libm-2.19.so
│   │   ├── libpcre.so.3 -> libpcre.so.3.13.1
│   │   ├── libpcre.so.3.13.1
│   │   ├── libpthread-2.19.so
│   │   ├── libpthread.so.0 -> libpthread-2.19.so
│   │   ├── librt-2.19.so
│   │   ├── librt.so.1 -> librt-2.19.so
│   │   ├── libselinux.so.1
│   │   ├── libtinfo.so.5 -> libtinfo.so.5.9
│   │   ├── libtinfo.so.5.9
│   │   ├── libz.so.1 -> libz.so.1.2.8
│   │   └── libz.so.1.2.8
│   ├── ld-2.19.so
│   └── ld-linux-aarch64.so.1 -> aarch64-linux-gnu/ld-2.19.so
│
└── usr
    ├── bin
    │   ├── cuobjdump
    │   ├── fatbinary
    │   ├── linux-sh
    │   ├── maxas
    │   ├── nvcc
    │   ├── nvcc_android
    │   ├── nvdisasm
    │   ├── nvlink
    │   └── ptxas
    ├── include
    │   └── wordexp.h
    ├── lib
    │   ├── aarch64-linux-gnu
    │   │   ├── libcloog-isl.so.4 -> libcloog-isl.so.4.0.0
    │   │   ├── libcloog-isl.so.4.0.0
    │   │   ├── libgmp.so.10 -> libgmp.so.10.1.3
    │   │   ├── libgmp.so.10.1.3
    │   │   ├── libisl.so.10 -> libisl.so.10.2.2
    │   │   ├── libisl.so.10.2.2
    │   │   ├── libmpc.so.3 -> libmpc.so.3.0.0
    │   │   ├── libmpc.so.3.0.0
    │   │   ├── libmpfr.so.4 -> libmpfr.so.4.1.2
    │   │   ├── libmpfr.so.4.1.2
    │   │   ├── libstdc++.so.6 -> libstdc++.so.6.0.19
    │   │   └── libstdc++.so.6.0.19
    │   ├── gcc
    │   │   └── aarch64-linux-gnu
    │   │       ├── 4.8
    │   │       │   ├── cc1
    │   │       │   ├── cc1plus
    │   │       │   ├── collect2
    │   │       │   ├── include
    │   │       │   │   ├── arm_neon.h
    │   │       │   │   ├── backtrace.h
    │   │       │   │   ├── backtrace-supported.h
    │   │       │   │   ├── float.h
    │   │       │   │   ├── iso646.h
    │   │       │   │   ├── omp.h
    │   │       │   │   ├── stdalign.h
    │   │       │   │   ├── stdarg.h
    │   │       │   │   ├── stdbool.h
    │   │       │   │   ├── stddef.h
    │   │       │   │   ├── stdfix.h
    │   │       │   │   ├── stdint-gcc.h
    │   │       │   │   ├── stdint.h
    │   │       │   │   ├── stdnoreturn.h
    │   │       │   │   ├── unwind.h
    │   │       │   │   └── varargs.h
    │   │       │   ├── liblto_plugin.so -> liblto_plugin.so.0.0.0
    │   │       │   ├── liblto_plugin.so.0 -> liblto_plugin.so.0.0.0
    │   │       │   ├── liblto_plugin.so.0.0.0
    │   │       │   ├── lto1
    │   │       │   └── lto-wrapper
    │   │       └── 4.8.2 -> 4.8/
    │   ├── libbfd-2.24-system.so
    │   └── libopcodes-2.24-system.so
    │
    ├── linux
    │   ├── ar
    │   ├── as
    │   ├── busybox
    │   ├── c++filt
    │   ├── cpp -> cpp-4.8
    │   ├── cpp-4.8
    │   ├── g++ -> g++-4.8
    │   ├── g++-4.8
    │   ├── gcc -> gcc-4.8
    │   ├── gcc-4.8
    │   ├── gcc-ar -> gcc-ar-4.8
    │   ├── gcc-ar-4.8
    │   ├── gcc-nm -> gcc-nm-4.8
    │   ├── gcc-nm-4.8
    │   ├── gcc-ranlib -> gcc-ranlib-4.8
    │   ├── gcc-ranlib-4.8
    │   ├── ld -> ld.bfd
    │   ├── ld.bfd
    │   ├── objcopy
    │   ├── objdump
    │   ├── readelf
    │   └── uname
    │
    ├── local
    │   ├── cuda -> cuda-7.0/
    │   └── cuda-7.0
    │       ├── bin
    │       │   ├── bin2c
    │       │   ├── crt
    │       │   │   ├── link.stub
    │       │   │   └── prelink.stub
    │       │   ├── cudafe
    │       │   ├── cudafe++
    │       │   ├── cuda-gdb
    │       │   ├── cuda-gdbserver
    │       │   ├── cuda-memcheck
    │       │   ├── cuobjdump
    │       │   ├── cuobjdump65
    │       │   ├── cuobjdump80
    │       │   ├── fatbinary
    │       │   ├── filehash
    │       │   ├── nvcc
    │       │   ├── nvcc.profile
    │       │   ├── nvdisasm
    │       │   ├── nvdisasm65
    │       │   ├── nvdisasm80
    │       │   ├── nvlink
    │       │   ├── nvprof
    │       │   ├── nvprune
    │       │   └── ptxas
    │       │
    │       ├── include (... include all subfolders)
    │       │
    │       ├── lib -> lib64/
    │       ├── lib64 (... include all subfolders)
    │       │
    │       ├── nvvm
    │       │   ├── bin
    │       │   │   └── cicc
    │       │   ├── include
    │       │   │   └── nvvm.h
    │       │   ├── lib64
    │       │   │   ├── libnvvm.so -> libnvvm.so.3
    │       │   │   ├── libnvvm.so.3 -> libnvvm.so.3.0.0
    │       │   │   └── libnvvm.so.3.0.0
    │       │   ├── libdevice
    │       │   │   ├── libdevice.compute_20.10.bc
    │       │   │   ├── libdevice.compute_30.10.bc
    │       │   │   └── libdevice.compute_35.10.bc
    │       │   └── libnvvm-samples (... include all subfolders)
    │       │
    │       └── samples (... include all subfolders)
    │
    └── tmp

```
