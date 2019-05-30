# Smart-Fluidnet
Smart-Fluidnet is a framework that automates model generation for fluid dynamic simulation. It is developed by PASA lab (http://pasa.ucmerced.edu/) at University of California, Merced. Smart-Fluidnet provides flexibility and generalization to automatically search best neural network(NN) models for different input problems.


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


