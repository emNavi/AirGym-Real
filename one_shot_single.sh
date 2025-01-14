#!/bin/bash
echo "[START] one_shot script with vins and control_for_gym"

if [ $? -eq 0 ] 
then
    echo "start camera"
    gnome-terminal -- bash -c "source devel/setup.bash;roslaunch vins rs_camera.launch" && sleep 3;
    echo "start ctrl_bridge"
    gnome-terminal -- bash -c "source devel/setup.bash;roslaunch control_for_gym ctrl_bridge.launch" && sleep 4;
    echo "start vins"
    gnome-terminal -- bash -c "source devel/setup.bash;roslaunch vins d435_no_rviz.launch" && sleep 5;
else
    echo error
    exit 
fi

echo "if you want to close this script, you need use : ./kill_one_shot.sh"