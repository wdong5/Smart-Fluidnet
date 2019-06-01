# Smart-Fluidnet
Smart-Fluidnet is a framework that automates model generation for fluid dynamic simulation. It is developed by PASA lab (http://pasa.ucmerced.edu/) at University of California, Merced. Smart-Fluidnet provides flexibility and generalization to automatically search the best neural network(NN) models for different input problems.


## Step 1: Installing mantaflow 

The first step is to download the custom manta fork.

`git clone git@github.com:kristofe/manta.git`

Next, you must build mantaflow using the cmake system.
```
cd FluidNet/manta
mkdir build
cd build
sudo apt-get install doxygen libglu1-mesa-dev mesa-common-dev qtdeclarative5-dev qml-module-qtquick-controls
cmake .. -DGUI='OFF' 
make -j8
```
## Step 2: generating input problems

We use a subset of the NTU 3D Model Database models (http://3d.csie.ntu.edu.tw/~dynamic/database/). Please download the model files:
```
cd FluidNet/voxelizer
mkdir objs
cd objs
wget http://3d.csie.ntu.edu.tw/~dynamic/database/NTU3D.v1_0-999.zip
# wget https://cs.nyu.edu/~schlacht/NTU3D.v1_0-999.zip  # Alternate download location.
unzip NTU3D.v1_0-999.zip
wget https://www.dropbox.com/sh/5f3t9abmzu8fbfx/AAAkzW9JkkDshyzuFV0fAIL3a/bunny.capped.obj
```
Next we use the binvox library (http://www.patrickmin.com/binvox/) to create voxelized representations of the NTU models. Download the executable for your platform and put the binvox executable file in FluidNet/voxelizer. Then run our script:

```
cd FluidNet/voxelizer
chmod u+x binvox
python generate_binvox_files.py
```
Install matlabnoise (https://github.com/jonathantompson/matlabnoise) to the SAME path that FluidNet is in. i.e. the directory structure should be:
```
/path/to/FluidNet/
/path/to/matlabnoise/
```
To install matlabnoise (with python bindings):
```
sudo apt-get install python3.5-dev
sudo apt-get install swig
git clone git@github.com:jonathantompson/matlabnoise.git
cd matlabnoise
sh compile_python3.5_unix.sh
sudo apt-get install python3-matplotlib
python3.5 test_python.py
```
Now you're ready to generate the training data. Make sure the directory data/datasets/output_current exists. 
```
cd FluidNet/manta/build
./manta ../scenes/_trainingData.py --dim 2 --addModelGeometry True --addSphereGeometry True
```
## Step3: generating models

RUNNING TORCH7 TRAINING

We assume that Torch7 is installed, otherwise follow the instructions here. We use the standard distro with the cuda SDK for cutorch and cunn and cudnn.

After install torch, compile tfluids:
```
sudo apt-get install freeglut3-dev
sudo apt-get install libxmu-dev libxi-dev
cd FluidNet/torch/tfluids
luarocks make tfluids-1-00.rockspec
```

Note: some users are reporting that you need to explicitly install findCUDA for tfluids to compile properly with CUDA 7.5 and above.
```
luarocks install findCUDA
```

To run the interactive demo firstly compile LuaGL:

```
git clone git@github.com:kristofe/LuaGL.git
cd LuaGL
luarocks make luagl-1-02.rockspec
```

##Step4: Running the Smart-Fluidnet
Dowload our trained models:
```
./download_models.sh
```

Run the script:
```
./runtime_algorithm.sh
```

