#!/bin/bash
echo "<Send Target>"

# kill all other target publisher
kill $(ps -aux|grep target_pub|grep -v grep|awk '{print $2}') && sleep 1

if [ $? -eq 0 ]
    echo command running
then
    gnome-terminal -- bash -c "source devel/setup.bash;roslaunch inference target_pub.launch";
else
    echo error
fi
