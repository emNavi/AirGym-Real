kill $(ps -aux|grep rosmaster|grep -v grep|awk '{print $2}')
kill $(ps -aux|grep px4.launch|grep -v grep|awk '{print $2}')
kill $(ps -aux|grep apm.launch|grep -v grep|awk '{print $2}')
kill $(ps -aux|grep d435|grep -v grep|awk '{print $2}')
kill $(ps -aux|grep realsense|grep -v grep|awk '{print $2}')
kill $(ps -aux|grep ctrl_bridge.launch|grep -v grep|awk '{print $2}')
kill $(ps -aux|grep target_pub|grep -v grep|awk '{print $2}') && sleep 1
kill $(ps -aux|grep inference|grep -v grep|awk '{print $2}') && sleep 1

