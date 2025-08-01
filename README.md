
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"


export CMAKE_ARGS="\
-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
-DUSE_CUDA=ON \
-DUSE_SYSTEM_NVTX=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH=$CUDA_HOME"

sudo python3 setup.py clean
sudo python3 setup.py install
