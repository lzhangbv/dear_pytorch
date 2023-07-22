# Errors Encountered When Configuring the Environment and Their Solutions

### 1. Error Message When Running "pip -r requirements.txt"

```
 error[E0557]: feature has been removed
    --> /home/hxyu/.cargo/registry/src/mirrors.tuna.tsinghua.edu.cn-df7c3c540f42cdbd/lock_api-0.3.4/src/lib.rs:91:42
     |
  91 | #![cfg_attr(feature = "nightly", feature(const_fn))]
     |                                          ^^^^^^^^ feature has been removed
     |
     = note: split into finer-grained feature gates
```

At first, we thought it was a problem with the Rust version, so we installed various versions of Rust and tried switching between them, but the error persisted. After troubleshooting, we found that the issue occurred during the installation of the transformers library. Installing the latest version of transformers did not cause any issues, but installing the specified version in requirements.txt resulted in the aforementioned error.

**Solution: The issue was caused by a Python environment version that was too high. Switching to Python 3.7 resolved the issue.**

### 2. Another Error Occurs When Running "pip -r requirements.txt"

```
error: Cannot link MPI programs . Check your configuration!!![end of output]
note: This error originates from a subprocess, and is likely not a problem with p
ERROR: Failed building wheel for mpi4py
```

**Solution: Install OpenMPI and specify the MPI path.**

#### Install OpenMPI (compiled from source code)

1. Download the source code:

   ```shell
   wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
   tar -xvf openmpi-4.1.5.tar.gz
   ```

2. Compile and install:

   ```sh
   # Installation path
   mkdir build
   cd openmpi-4.1.5
   # /home/yourpath/build: installation path, /usr/local/cuda: CUDA location, which cuda can be used to find it
   ./configure --prefix=/home/yourpath/build --with-cuda=/usr/local/cuda
   
   make -j20
   make install
   ```

3. Modify environment variables, which can be directly written into bashrc:

   ```sh
   MPI_HOME=/home/yourpath/build
   export PATH=${MPI_HOME}/bin:$PATH
   export LD_LIBRARY_PATH=${MPI_HOME}/lib:/usr/local/lib
   export MANPATH=${MPI_HOME}/share/man:$MANPATH
   ```

4. Test:

   ```
   cd ~/openmpi-4.1.5/examples
   make
   mpirun -np 4 hello_c
   ```

5. Install mpi4py:

   ```sh
   conda uninstall mpich
   # /home/yourpath/build/bin/mpicc: mpicc path, which mpicc can be used to find it
   env MPICC=/home/yourpath/build/bin/mpicc pip -v --no-cache-dir install --no-binary :all: mpi4py==3.1.1
   
   ```

   

### 3. Install NCCL without Root Privileges

When installing Horovod, the following error message appears.

```shell
CMake Error at /usr/share/cmake-3.16/Modules/FindPackageHandleStandardArgs .cmake:146 (message):
	Could NOT find NCCL (missing: NCCL LIBRARY)Call Stack (most recent call first):/usr/share/cmake-3.16/Modules/FindPackageHandleStandardArgs .cmake:393( FPHSA FAILURE MESSAGE)cmake/Modules/FindNCCL.cmake:42 (find package handle standard args)CMakeLists .txt:195 (find package)
Configuring incomplete, errors occurred!See also "/tmp/pip-install-k5kpnjie/horovod blef98631ecc48eba0cf09c7c805a25f/build/temp.linux-x86 64-cpython-37/CMakeFiles/CMake0utput.log"
```

**Solution:NCCL is a prerequisite for installing horovod.**

#### Method 1 (Cannot Specify Installation Version): Refer to https://github.com/NVIDIA/nccl

1. Download the NCCL project to the server:

   ```sh
   git clone git@github.com:NVIDIA/nccl.git
   ```

2. Compile and install:

   ```shell
   cd nccl
   make -j20 src.build
   ```

3. Modify environment variables:

   ```sh
   vi ~/.bashrc
   ```

4. Add the following line to the end of the file:

   ```shell
   export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH
   /path/to/nccl/lib: path where NCCL is installed.
   ```

5. Save the file and execute:

   ```shell
   source ~/.bahsrc
   ```

   

#### Method 2 (Specify Version):

1. Visit [NVIDIA Collective Communications Library (NCCL) Legacy Download Page | NVIDIA Developer](https://developer.nvidia.com/nccl/nccl-legacy-downloads) to download the NCCL O/S agnostic local installer for the corresponding CUDA version. The version downloaded on this machine is cuda10.2+nccl=2.6.4.

2. Upload the compressed txz file to the server.

3. Unzip:

   ```shell
   mkdir nccl
   tar -xvf your_file.txz -C ./nccl
   ```

4. Modify environment variables, refer to steps 3~5 in Method 1.