#!/bin/bash
echo "<Takeoff>"

if [ $? -eq 0 ] 
    echo "command running"
then
    rostopic pub /swarm_takeoff std_msgs/Float32MultiArray "
    layout: 
        dim: 
            - 
                label: ''
                size: 0
                stride: 0
        data_offset: 0
    data: [1, 1]
    "  --once
else
    echo error
fi

echo "Drone Takeoff"
