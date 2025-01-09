#!/home/emnavi/miniconda3/envs/inference/bin/python

import rospy
import random
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

def target_pub(target_state):
    # Create the publisher
    pub = rospy.Publisher('/target_state', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz

    # Set target state
    # matrix(0-9);pos(9-12);vel(12-15);rate(15-18)
    state = Float64MultiArray()
    state.data = target_state
    
    # Publish the message
    while not rospy.is_shutdown():
        pub.publish(state)
        # rospy.loginfo(state.data)
        rate.sleep()

if __name__ == '__main__':
    # Set the node name
    rospy.init_node('target_pub', anonymous=True)
    # 获取target_state参数，参数是一个字符串，需要转换为浮点数列表
    target_state = rospy.get_param('~target_state', [])
    rospy.loginfo("Received target_state: {}".format(target_state))

    target_pub(target_state)