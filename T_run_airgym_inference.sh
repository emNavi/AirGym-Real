#!/bin/bash
echo "<AirGym Inference>"

if [ $? -eq 0 ]
then
    gnome-terminal -- bash -c "source devel/setup.bash;roslaunch inference inference_runner.launch";
else
    echo error
fi
