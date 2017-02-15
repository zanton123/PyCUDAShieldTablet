# Programming and Running PyCUDA applications on the Google Pixel C (Tegra X1) and NVIDIA Shield Tablet (Tegra K1) using CUDA 7.0

This repository contains the files to get PyCUDA running on the Google Pixel C using Termux (by Fredrik Fornwall). Instructions are usable as of now but some improvements are pending.

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

Essentially, this is the same idea as used by the GNURoot Debian app but we make very limited use of the PRoot facilities.

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

The NVIDIA CUDA for Android files are now in /data/data/com.termux/files/usr/local/cuda-7.0. Make sure that files in the bin/ folder have execute permission. We will install the Linux for Tegra executables for compiling CUDA C++ next.

##NVIDIA Linux for Tegra

NVIDIA does not supply CUDA compiler components that run natively on Android (the toolkit contains only profiling and analysis tools ported to Android). However, both armhf for the Tegra K1 and ARM64 for the Tegra X1 versions are in the Linux for Tegra Archive: 

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

**_For the Pixel C:_**

From the **Tegra_Linux_Sample-Root-Filesystem_R24.1.0_aarch64.tbz2** archive copy the following files below from the **/lib/aarch64-linux-gnu/** to a portable drive. The CUDA toolkit files are in the **/usr/local/cuda-7.0/** folder of the **cuda-repo-l4t-7-0-local_7.0-76_arm64.deb** archive and need be transfered to the Termux usr/local/cuda-7.0/ folder. Copy the files, symbolic links and folders to the Pixel C establishing the following folder structure under the /data/data/com.termux/files/ Termux main folder:
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
Shell files for the home/ folder (cuda, normal, and .bashrc), and the usr/bin/ folder (nvcc, nvcc_android, and linux_sh) are in the respective folders of this repository. Make sure that files in bin/, usr/bin/, usr/linux/, usr/local/cuda-7.0/bin, and usr/local/cuda-7.0/nvvm/bin/ have execute permission.

Install CUDA 7.0 samples

```
tar -czvf CUDA-7.0-samples.tar.gz NVIDIA-cuda-7.0-samples

extract:

tar -xzvf <archive,tar.gz> <folder>

cd to folder, . cuda, cp nvcc_ ... script and compile . nvcc_ ...
```

##Install PyCUDA form sources

Download pycuda from PyPi https://pypi.python.org/pypi/pycuda the archive pycuda-2016.1.2.tar.gz to the Download folder using your web browser. For downloading using wget use the following URL for the file:

https://pypi.python.org/packages/e8/3d/4b6b622d8a22cace237abad661a85b289c6f0803ccfa3d2386103307713c/pycuda-2016.1.2.tar.gz

```
cd
mkdir pycuda
cd pycuda
cp ~/storage/shared/Download/pycuda-2016.1.2.tar.gz ./
tar -xzvf pycuda-2016.1.2.tar.gz
cd  pycuda-2016.1.2
```

On the Shield Tablet you will need to set C and C++ compiler flags for GNU make (not needed on the Pixel C):
```
export CFLAGS="-target=armv7-linux-android -D__ANDROID__"
export CXXFLAGS="-target=armv7-linux-android -D__ANDROID__ -stdlib=libgnustl_shared"
# else you will generate a dynamic link error for missing __atomic_fetch_add_4 symbol

python2 configure.py --cuda-root=/data/data/com.termux/files/usr/local/cuda
make

make install
```
On the Shield Tablet the nvcc shell script needs to option nvcc as follows for pycuda to run (not needed on the Pixel C):
proot ... /nvcc -std=c++11 -m32 -Xcompiler -mandroid -Xcompiler -D__ANDROID__
Do not specify the compute -arch=sm_32 here as pycuda will do so and nvcc fails on a duplicate compute arch option (pycuda uses -ptx, -fatbin, or -cubin compilation modes).
```
cd test
python2 test_driver.py
python2 test_gpuarray.py
python2 test_cumath.py
```
Examine the output and check the errors. Some test will fail on the tegras. However, if there is ubandant fail something is probably be wrong.
On the Pixel C (Tegra X1), test_driver.py fails the test_registered_memory and streamed_kernels tests, test_cumath.py fails the test_unary_func_kwargs test, and test_gpuarray.py fails test_scan.
On the Shield Tablet (Tegra K1), test_driver.py fails the test_simple_kernel_2 (python int too large to convert to C long), and test_register_host_memory (function not supported) tests, test_cumath.py passes all 28 tests, and test_gpuarray.py fails test_scan (import error on Mako). Overall the tegra K1 seems to perform pretty well despite we are using unsupported CUDA 7.0 from the Tegra X1 Jetpack 2.2 (Tegra K1 Jetpack has CUDA 6.5). Care should be taken with integer data types on the 32-bit system.

In case you want to rebuild pycuda with different options in the future the following packages should be uninstalled:
```
pip2 uninstall pycuda
pip2 uninstall pytools
```

##Installing MAXAS on the Pixel C

MAXAS by Scott Grey at NervanaSystems https://github.com/NervanaSystems/maxas requires nvdisasm from the CUDA 6.5 toolkit (maxas invokes cuobjdump, which then invokes nvdisasm). For the Pixel C we will require a arm64 version for Tegra X1. Nvidia offers a generic CUDA 6.5 toolkit for arm64 in the CUDA archive: https://developer.nvidia.com/cuda-toolkit-65

Download the generic CUDA Toolkit under ARMv8 64-bit*** and extract its content. When asked by the installer provide a path in your Download folder, eg ~/Downloads/cuda65arm64
```
chmod +x  cuda_6.5.14_linux_aarch64_native.run
./cuda_6.5.14_linux_aarch64_native.run --extract-only ~/Downloads/cuda65arm64
```
Once the run file archive is unpacked, nvdisasm and cuobjdump executables are in ~/Downloads/cuda65arm64/bin. Copy this file to the Pixel C Downloads folder and them into the cuda/bin folder next to the CUDA 70 versions:
```
cd
cp ~/storage/shared/Downloads/cuobjdump ../usr/local/cuda/bin/nvdisasm65
cp ~/storage/shared/Downloads/cuobjdump ../usr/local/cuda/bin/cuobjdump65
chmod +x ../usr/local/cuda/bin/nvdisasm65
chmod +x ../usr/local/cuda/bin/cuobjdump65
```

Install maxas from github repository https://github.com/NervanaSystems/maxas.git
```
cd
git clone https://github.com/NervanaSystems/maxas.git
cd maxas
perl Makefile.PL
make
make install
```

Generate scripts for running the cuobject and nvdiasm linux binaries under Android and conveniently engaging maxas:
```
cd
vi ../usr/bin/maxas
```
Enter insert mode (`i`) and type:
```
maxas.pl $@
```

Save and close the file with `Ctrl`+`c`, then `:wq` ENTER, then set executable permission:
```
chmod +x ../usr/bin/maxas
vi ../usr/bin/nvdisasm
```
Enter insert mode (`i`) and type:
```
#! /data/data/com.termux/files/usr/bin/sh
/data/data/com.termux/files/lib/ld-linux-aarch64.so.1 /data/data/com.termux/files/usr/local/cuda/bin/nvdisasm65 $@
```

Save and close the file with `Ctrl`+`c`, then `:wq` ENTER, then set executable permission:
```
chmod +x ../usr/bin/nvdisasm
vi ../usr/bin/cuobjdump
```
Enter insert mode (`i`) and type:
```
#! /data/data/com.termux/files/usr/bin/sh
/data/data/com.termux/files/lib/ld-linux-aarch64.so.1 /data/data/com.termux/files/usr/local/cuda/bin/cuobjdump65 $@
```

Save and close the file with `Ctrl`+`c`, then `:wq` ENTER, then set executable permission:
```
chmod +x ../usr/bin/cuobjdump
```
add to the LD_LIBRARY_PATH
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/data/com.termux/files/lib/aarch64-linux-gnu:/data/data/com.termux/usr/lib/aarch64-linux-gnu
```
Try it out:
```
cd
cd CUDA-7.0-samples/0_Simple/vectorAddDrv
nvcc -cubin -arch=sm_50 vectorAdd_kernel.cu -o vectorAdd_kernel_50.cubin
maxas -t vectorAdd_kernel_50.cubin
```

The -t command test if instructions in the cubin can be decoded. You will see some warnings about deprecated perl regex functions but no error. To extract a SASS file use the following command:
```
maxas -e vectorAdd_kernel_50.cubin vectorAdd_kernel_50.sass
cat vectorAdd_kernel_50.sass
```

You should see now the SASS of the vectorAdd_kernel. Note that this all is for compute architecture sm_50 as nvdisasm from the CUDA toolkit 6.5 cannot process sm_53. This is a problem if you want to assemble the kernel and run it on the Pixel C. Tegra X1 requires sm53 cubin code and will simply not run sm_50 code. One way to circumvent this problem is by reinserting the (modified) kernel into a sm_53 cubin file that is compiled by nvcc from the same vectorAdd_kernel.cu. The following instructions will insert the SASS into the vectorAdd_kernel.cubin file in a way that it can run on the Pixel C:
```
nvcc -cubin -arch=sm_53 vectorAdd_kernel.cu -o vectorAdd_kernel_53.cubin
maxas -i vectorAdd_kernel_50.sass vectorAdd_kernel_53.cubin vectorAdd_kernel.cubin
```
To test you will need to build the vectorAddDrv.cpp executable. This can be compiled using clang as it is a regular c++ file (you will need to link with libcuda.so). The following instructions use nvcc to show that it works:
```
nvcc -m64 -std=c++11 -Xcompiler -D__ANDROID__ -Xcompiler -mandroid -Xcompiler -fPIC -Xcompiler -pie -Xcompiler -nostdlib -Xlinker -nostdlib -Xlinker --eh-frame-hdr -Xlinker -pie -Xlinker -maarch64linux -Xlinker -dynamic-linker=/system/bin/linker64 -arch=sm_53 -I/usr/lib/gcc/aarch64-linux-gnu/4.8/include -I/usr/local/cuda/include -I../../common/inc -Xlinker /usr/lib/crtbegin_dynamic.o -L/system/lib64 -L/system/vendor/lib64 -lnvcompute -L/usr/local/cuda/lib64 -lcudart -L/usr/lib -lgnustl_shared -Xlinker /usr/lib/crtend_android.o -o vectorAddDrv vectorAddDrv.cpp
```

This command line can be adapted to compile any cuda application. And most of the options are necessary. To make it easier you can generate a nvcc_android script that contains all options but the source and output file names. Now run the vectorAddDrv executable:
```
./vectorAddDrv
```
That's it! You just ran a kernel that has been reconstituted from SASS. Make sure you do not have any PTX files in the same folder. vectorAddDrv will first look for vectorAdd_kernel64.ptx and only if not found try to load vectorAdd_kernel.cubin. MaxAs is a very powerful and experimental tool so be careful you might mess up the control instructions and then it is up to your Tegra to survive!

Endnote: You can load (modified) cubin kernels in PyCUDA. The last lines of the maxas perl script instruct on how to insert modified kernels back into CUDA applications so they can be engaged using the CUDA runtime API. This makes the Pixel C a full-fledged CUDA development system. The latest MaxAsGrammar.pm adds capabilities for encoding Pascal instructions, which are GP104 and GP100 native. For using nvdisasm from CUDA 8.0 to generate SASS from sm_60 kernels MaxAs.pm has been adapted by Hugh Perkins and is available at:
 https://github.com/hughperkins/maxas/blob/1c0fef7298152e9e62ca811a7d4ccba30541fcd3/lib/MaxAs/MaxAs.pm

Essentially, this allows you to write SASS for kernels running on Piz Daint. You will need to get a good grip on SASS and the GPU architecture, though.
