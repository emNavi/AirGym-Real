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

## Calibration
Calibration is needed before use because all the state estimation comes from vins_fusion. Steps below is calobration for `vins_fusion`.

Create an output directory at `~`:
```bash
mkdir ~/vins_output
```

Find `vins_fusion_d435/param_files/real_use/vins/vins_with_d435.yaml`, edit `estimate_extrinsic` to 1 before run `one_shot_single.sh`
```
# Extrinsic parameter between IMU and Camera.
estimate_extrinsic: 0  # 0  Have an accurate extrinsic parameters. We will trust the following imu^R_cam, imu^T_cam, don't change it.
                        # 1  Have an initial guess about extrinsic parameters. We will optimize around your initial guess.
```
then run `one_shot_single.sh` to launch camera and vins. 

After initialization, slowly pick up the drone by hand and move it through a textured area at a low speed (<1 m/s). Typically, you can walk around the area in a circle and return to the starting point. Observe the output of the vision odometry on the ROS topic `/mavros/vision_pose/pose.pose.position`. If the resulting values are close to the initial values (x: 0, y: 0, z: 0) with deviations within the centimeter range, it indicates that Vins-Fusion is operating correctly.

Use parameters generated in `~/vins_output` and copy it to replace corresponding part in `vins_fusion_d435/param_files/real_use/vins/vins_with_d435.yaml`, for example:
```bash
body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ 1.4891044568646539e-02, 6.8362959335264793e-03,
       9.9986575191350757e-01, 3.7840729852448969e-02,
       -9.9975608165935537e-01, -1.6210178102668449e-02,
       1.5000243698423274e-02, 2.0274150716127610e-02,
       1.6310548022273979e-02, -9.9984523561588712e-01,
       6.5932419509737716e-03, 4.1220449189954857e-02, 0., 0., 0., 1. ]
body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ 1.5289913963302260e-02, 7.2208531242852298e-03,
       9.9985702868517767e-01, 3.1072342493410010e-02,
       -9.9975052829224142e-01, -1.6172957303737467e-02,
       1.5405084596829592e-02, -2.9047067859331231e-02,
       1.6281882888008119e-02, -9.9984313506280009e-01,
       6.9717686000185797e-03, 4.2984672415576880e-02, 0., 0., 0., 1. ]
```
> Note: After the calibration, must remenber to edit `estimate_extrinsic` back to 0!