# DeFoP: Deep Forest Pilot

This work is based on the paper "Robust Autonomous Navigation of Aerial Robots in Dense Forests Using Learned Motion Evaluation, Depth Refinement, and Real-Time Supervisory Safety Mechanisms"

Please consider that this work is build on top of [sevae-ORACLE](https://github.com/ntnu-arl/ORACLE/) that you can check for more details about this work.

The VAE code can be found in [this repo](https://github.com/ntnu-arl/sevae).

## 1) Setup simulation environment

## Setup

```bash
mkdir -p defop_ws/src && cd defop_ws/src
git clone https://github.com/guglielmo610/DeFoP
```

## Build

```bash
sudo apt-get install ros-${ROS_DISTRO}-octomap-msgs ros-${ROS_DISTRO}-octomap-ros
cd defop_ws
catkin config -DCMAKE_BUILD_TYPE=Release --blacklist deep_collision_predictor rotors_hil_interface
catkin build
```

You also need to install the NVIDIA GPU driver, `CUDA toolkit`, and `cudnn` to run Tensorflow on NVIDIA GPU. A typical procedure to install them can be found in [Link](https://medium.com/@pydoni/how-to-install-cuda-11-4-cudnn-8-2-opencv-4-5-on-ubuntu-20-04-65c4aa415a7b), note that the exact versions may change depending on your system.

Additionally, create a conda environment:
```
# Follow the below procedure for CUDA 11+
conda create -n oracle_env python=3.8 libffi=3.3
conda activate oracle_env
cd defop_ws/src/planning/ORACLE/
pip install -r requirements_cuda11.txt 
```

Install additional Python packages in `oracle_env` from [seVAE repo](https://github.com/ntnu-arl/sevae)
```
conda activate oracle_env
cd defop_ws/src/planning/sevae
pip3 install -e .
```

If you are using ROS version < Noetic, then you need to build geometry, geometry2, and vision_opencv packages with python 3 (for `import tf, cv_bridge`) following the below instructions. 

### Build geometry, geometry2 and vision_opencv with python 3 (NO need for ROS Noetic)
First, we need to get the path to our conda env:
```
conda activate oracle_env
which python
```
You should get something like this `PATH_TO_YOUR_ORACLE_ENV/bin/python`. 

Then run the following commands (replace `PATH_TO_YOUR_ORACLE_ENV` with what you get in your terminal) to create and build a workspace containing geometry, geometry2, and vision_opencv packages:
```
mkdir ros_stuff_ws && cd ros_stuff_ws
mkdir src && cd src
git clone https://github.com/ros/geometry.git -b 1.12.0
git clone https://github.com/ros/geometry2.git -b 0.6.5
git clone https://github.com/ros-perception/vision_opencv.git -b YOUR_ROS_VERSION
catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=PATH_TO_YOUR_ORACLE_ENV/bin/python -DPYTHON_INCLUDE_DIR=PATH_TO_YOUR_ORACLE_ENV/include/python3.8 -DPYTHON_LIBRARY=PATH_TO_YOUR_ORACLE_ENV/lib/libpython3.8.so
catkin config --install
catkin build geometry geometry2 vision_opencv -DSETUPTOOLS_DEB_LAYOUT=OFF
```

Don't forget to source this folder in your terminal
```
source ros_stuff_ws/devel/setup.bash
# or source ros_stuff_ws/install/setup.bash --extend
```

## 2) Generate training data: 

Set `EVALUATE_MODE = False` and `RUN_IN_SIM = True` in `config.py` file.

Run in one terminal (NOT in `conda` virtual environment)
```
roslaunch rmf_sim rmf_sim_sevae.launch
```

Open another terminal, source `defop_ws` workspace and run inside `deep_collision_predictor` folder (**Note**: remember to set `PLANNING_TYPE=1` in `config.py`)
```
# conda activate oracle_env
python generate/generate_data_info_gain.py --save_path=path_to_folder
``` 

If `--save_path` is not specified, the default path in `common_flags.py` is used.

## 3) Process the training data:

Run the script in seVAE [repo](https://github.com/ntnu-arl/sevae) to create the `di_latent.p` and `di_flipped_latent.p` pickle files. Put the latent pickles in the same folder as the other pickle files in step 2 above.

Then run
```
# conda activate oracle_env
python process/data_processing_sevae.py --load_path=path_to_folder --save_tf_path=path_to_folder
```

If `--load_path` or `--save_tf_path` is not specified, the default path in `common_flags.py` is used. \
The tfrecord files created from `data_processing.py` are saved in `save_tf_path`. \
Split the tfrecord files into 2 folders for training and validation (80/20 ratio).

## 4) Train the collision prediction network:

```
# conda activate oracle_env
python train/training.py --training_type=1 --train_tf_folder=path_to_folder --validate_tf_folder=path_to_folder --model_save_path=path_to_folder
```

If `--train_tf_folder` or `--validate_tf_folder` or `--model_save_path` is not specified, the default path in `common_flags.py` is used.

## 5) Optimize the network for inference speed (with TensorRT, optional)

**Note**: For multi-GPU systems, you may need to `export CUDA_VISIBLE_DEVICES=0` to run TensorRT, otherwise you can get some runtime errors.

Set the path to the .hdf5 file using `--checkpoint_path` when calling python scripts in `optimize` folder. The resulting .trt or .onnx files will be created in the main folder of this package.

```
# conda activate oracle_env
python3 optimize/convert_keras_combiner_tensorrt_engine_sevae.py --checkpoint_path=PATH_TO_HDF5_FILE
python3 optimize/convert_keras_rnn_to_tensorrt_engine_sevae.py --checkpoint_path=PATH_TO_HDF5_FILE
```

## 6) Evaluate the planner

Choose `PLANNING_TYPE` in `config.py` file (for evaluating A-ORACLE in sim, enable the RGB camera xacro in `rmf_sim/rmf_sim/rotors/urdf/delta.gazebo`)

If using Tensorflow model for inference, set `COLLISION_USE_TENSORRT = False` or `INFOGAIN_USE_TENSORRT = False` in `config.py` file and update the path to the weight files (.hdf5 files) in `config.py`.

If using TensorRT model for inference, set `COLLISION_USE_TENSORRT = True` or `INFOGAIN_USE_TENSORRT = True` in `config.py` file and update the path to the weight folders (containing .trt files) in `config.py`. **Note**: for multi-GPU systems, you may need to `export CUDA_VISIBLE_DEVICES=0` to run TensorRT, otherwise you can get some runtime errors.

Change the `world_file` argument in `rmf_sim.launch` to choose the testing environment. We provide some testing environments in `rmf_sim/worlds` folder. Additionally, set `rviz_en` to `true` in `rmf_sim.launch` for visualization of the network's prediction. Please refer to the [wiki](https://github.com/ntnu-arl/ORACLE/wiki) present in the ORACLE project for detailed instructions to run the demo simulations as well as documentation of parameters in `config.py`.

### In SIM

## Configuration

```bash
# In the config.py file, set the following parameters
PLANNING_TYPE = 1
EVALUATE_MODE = True
RUN_IN_SIM = True

# If using CPN's Tensorflow model for inference:
# Disable TensorRT
COLLISION_USE_TENSORRT = False
# Update the path to the weight files (.hdf5)
seVAE_CPN_TF_CHECKPOINT_PATH = "PATH_TO_HDF5_FILES"

# If using CPN's TensorRT model for inference (recommended for faster speed):
# Enable TensorRT
COLLISION_USE_TENSORRT = True
# Update the path to the TensorRT weight folders (.trt files)
seVAE_CPN_TRT_CHECKPOINT_PATH = "PATH_TO_TRT_FOLDER"
```
Example config files provided:

- `config_gazebo_corridor_sevae_oracle_naive.py`  
  *(TensorFlow inference, no Ensembles, no Unscented Transform)*  
- `config_gazebo_corridor_sevae_oracle.py`  
  *(TensorRT inference, Deep Ensembles, Unscented Transform)*  

These can be found in the `config_files` folder.  
You can copy-paste their contents into your `config.py`.

The best results in simulation have been obtained with these configuration of weights:

"seVAE_CPN_TF_CHECKPOINT_PATH = ['model_weights/VAE_EVO_back_to_origin/net1/saved-model.hdf5', # SKIP_STEP_GENERATE = 5, ACTION_HORIZON = 15, VEL_MAX = 2.0 m/s
                        'models/4/saved-model-70.hdf5',
                        'models/4/saved-model-20.hdf5'"

where the first network was a pretrained network present already available in sevae-ORACLE and the other 2 were models that were fined tuned in this work in a simulated forest enviornment.
All these pretrained weights are available in the folder "models"

---

## Run Simulation

### Terminal 1 (NOT inside a conda environment)

```bash
roslaunch rmf_sim rmf_sim.launch
```

### Terminal 2: Run the seVAE node

```bash
# conda activate oracle_env
cd PATH_TO_lmf_sim_ws/src/planning/sevae/sevae/inference/src
python vae_node.py --sim=True
```

### Terminal 3: Run the CPN node

```bash
# conda activate oracle_env
source PATH_TO_defop_ws/devel/setup.bash
source PATH_TO_ros_stuff_ws/devel/setup.bash   # only if ROS version < Noetic
python evaluate/evaluate.py
```

---

## Start the Planner

Wait until you see:

```
START planner
```

printed in the second terminal, then call:

```bash
rosservice call /start_planner "{}"
```

---

## Notes

You can visualize the reconstructed image from the seVAE decoder in RViz by subscribing to:

```
/decoded_image
```


### In the real robot (see more info about the robot [here](https://github.com/ntnu-arl/lmf_gazebo_models))

## Configuration to use all the optimized features of DeepForestPilot (based on sevae-ORACLE)

```bash
# In the config.py file, set the following parameters
PLANNING_TYPE = 1
EVALUATE_MODE = True
RUN_IN_SIM = False
COLLISION_USE_TENSORRT = True
seVAE_CPN_TRT_CHECKPOINT_PATH = "PATH_TO_TRT_FOLDER"
```

The best results in real world have been obtained with these configuration of weights:

"seVAE_CPN_TRT_CHECKPOINT_PATH = ['model_weights/VAE_EVO_back_to_origin/nett5',
                                'model_weights/VAE_EVO_back_to_origin/nett5',
                                'model_weights/VAE_EVO_back_to_origin/nett6'"

where both the networks used were fined tuned in this work in a simulated forest enviornment and then converted in tensorRT for optimized results.
All these pretrained weights are available in the folder "model_weights"

---

## Run real-time sensors

### Terminal 1 (NOT inside a conda environment)

Here you have to run your autopilot, your camera and your odometry (in my case I generated a launch file that launch together my px4 controller, a realsense d435i and Vins Fusion, together with a bridge between Vins Fusion and the px4 to send the odometry in the Kalman Filter of the autopilot)

## Run real-time softwares

### Terminal 2: Run the real-time depth improver node

```bash
conda activate oracle_env
cd PATH_TO_defop_ws/
python img_filler.py
```

Inside img_filler.py set the parameters of depth image ros topic "self.sub" and threshold distance "close_pixels"

### Terminal 3: Run the seVAE node

```bash
conda activate oracle_env
source PATH_TO_defop_ws/devel/setup.bash
cd PATH_TO_defop_ws/src/planning/sevae/sevae/inference/src
python vae_node.py --sim=True
```

### Terminal 4: Run the CPN node

```bash
conda activate oracle_env
source PATH_TO_defop_ws/devel/setup.bash
cd PATH_TO_defop_ws/src/planning/ORACLE
python evaluate/evaluate.py
```
This file contains the main modifications introduced on top of the original **seVAE_ORACLE** implementation.

- **`slowdown_action`**  
  Detects indecision in the planner and commands the drone to briefly stop (1 second) to stabilize before continuing.

- **`deadend_action`**  
  A refined behavior for handling dead-end situations. When a dead end is detected, the drone rotates **away from the closest obstacle**, improving its ability to escape narrow or cluttered areas.

- **`filter_actions_by_grid_3`**  
  This is the **supervisory layer** on top of the neural network path planner. It filters out any motion primitives that would lead to an imminent collision.

  You can tune the following parameters to adjust the safety constraints:
  - `min_dist` — minimum allowed distance to obstacles  
  - `min_ratio` — occupancy ratio threshold defining how much of a sector must be filled before it is considered unsafe  
  - `drone_radius` — effective size (with margin) of your drone used for collision checks  

These additions significantly improve the robustness and safety of the planner when navigating dense forest environments.


---

## Start the Planner

Wait until you see:

```
START planner
```

printed in the second terminal, then call:

```bash
rosservice call /start_planner "{}"
```

---

## References

If you use this work in your research, please cite the following publications:


**Autonomous Drone Navigation in Forest Environments Using Deep Learning**

```
@Article{DelCol2025Autonomous,
AUTHOR = {Del Col, G. and Karjalainen, V. and Hakala, T. and Honkavaara, E.},
TITLE = {Autonomous Drone Navigation in Forest Environments Using Deep Learning},
JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {XLVIII-2/W11-2025},
YEAR = {2025},
PAGES = {87--93},
URL = {https://isprs-archives.copernicus.org/articles/XLVIII-2-W11-2025/87/2025/},
DOI = {10.5194/isprs-archives-XLVIII-2-W11-2025-87-2025}
}
```

**Robust Autonomous Navigation of Aerial Robots in Dense Forests Using Learned Motion Evaluation, Depth Refinement, and Real-Time Supervisory Safety Mechanisms**

```
Publication ongoing
```

## Contact

You can contact me for any question:
* [Guglielmo Del Col](guglielmodelcol@gmail.com)
