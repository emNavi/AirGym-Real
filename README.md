# AirGym-Real
AirGym-Real is a toolkit for quadrotor robot learning deployment, it is designed to be used with AirGym(https://github.com/emNavi/AirGym). 

**AirGym-Real** is an onboard ROS inference Node. It is implemented as a ROS node to obtain state from PX4 autopilot and RealSense D430 camera, as well as to load the sim-to-real policy model and conduct the inference.

## Features
- **Highly integrated**: AirGym-Real integrates functionalities including (1) neural network load and inference (2) sensing data subscribing and preprocess (3) computed actions publishing. Also, to support using in GNSS/positioning deny environment, AirGym-Real runs a VINS-fusion onboard to get state estimation; to communicate with MAVROS, it involves toolkit **control_for_gym** as submodule, to provide a FSM for flexible switching between tradition control and NN inference control.
- **Oneshot operation**: AirGym-Real encapsulates funstions into shells, to realize oneshot start and oneshot kill.

## Usage
### Installation
```bash
git clone git@github.com:emNavi/AirGym-Real.git
cd AirGym-Real
```
Create conda environment and compile. Note that conda environment must be activated before `catkin_make`.
```bash
./create_inference_conda_env.sh
conda activate inference
catkin_make
```

### Run
- **one_shot_single.sh**: launch camera, run control_bridge, run vins_fusion in one shot.
- **inference.sh**: launch inference node for neural network computing.
- **takeoff.sh**: execute quadrotor takeoff to a predefined height which can be edited in ctrl_bridge.launch
- **land.sh**: land.
- **send_target_state.sh**: send expected drone state.
- **rosbag.sh**: record selected rostopics.
- **kill_one_shot.sh**: kill all related pid.