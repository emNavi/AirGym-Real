#!/home/emnavi/miniconda3/envs/inference/bin/python

from airgym.lib.agent.players import BasePlayer, A2CPlayer
from airgym.lib.core.running_mean_std import RunningMeanStd
from airgym.lib.utils.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
import time

import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

from src.inference.src.utils import torch_ext

FRAME_LOCAL_NED=1
FRAME_LOCAL_OFFSET_NED=7
FRAME_BODY_NED=8
FRAME_BODY_OFFSET_NED=9

# for PositionTarget
IGNORE_PX=1
IGNORE_PY=2
IGNORE_PZ=4
IGNORE_VX=8
IGNORE_VY=16
IGNORE_VZ=32
IGNORE_AFX=64
IGNORE_AFY=128
IGNORE_AFZ=256
FORCE=512
IGNORE_YAW=1024
IGNORE_YAW_RATE=2048
# for AttitudeTarget
IGNORE_ROLL_RATE=1
IGNORE_PITCH_RATE=2
IGNORE_YAW_RATE_=4
IGNORE_THRUST=64
IGNORE_ATTITUDE=128

def quaternion_to_matrix(quat: torch.Tensor):
    """Convert a quaternion to a rotation matrix.
    
    Args:
        quat (Tensor): a tensor of shape (4,) representing the quaternion (w, x, y, z).
    
    Returns:
        Tensor: a tensor of shape (3, 3) representing the rotation matrix.
    """
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    matrix = torch.tensor([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ], device=quat.device)
    
    return matrix

class CpuPlayerContinuous(A2CPlayer):
    def __init__(self, params):
        super().__init__(params)
        print("Running on", self.device)

        # initialize
        rospy.init_node('onboard_computing_node', anonymous=True)

        self.target_state = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], device=self.device)

        self.obs_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.callback)
        self.image_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_cb)
        self.bridge = CvBridge()
        # self.obs_sub = rospy.Subscriber('/vins_fusion/odometry', Odometry, self.callback)
        self.target_sub = rospy.Subscriber('/target_state', Float64MultiArray, self._callback)
        ctl_mode = self.ctl_mode = self.env_config.get('ctl_mode')

        self.action_pub = rospy.Publisher('/airgym/inference_result', Float64MultiArray, queue_size=2000)
        
        if ctl_mode == "pos":
            self.cmd_pub = rospy.Publisher('/airgym/cmd', PositionTarget, queue_size=2000)
            self.action = torch.zeros(4, device=self.device)
        elif ctl_mode == "vel":
            self.cmd_pub = rospy.Publisher('/airgym/cmd', Twist, queue_size=2000)
            self.action = torch.zeros(4, device=self.device)
        elif ctl_mode == "atti":
            self.cmd_pub = rospy.Publisher('/airgym/cmd', AttitudeTarget, queue_size=2000)
            self.action = torch.zeros(5, device=self.device)
        elif ctl_mode == "rate":
            self.cmd_pub = rospy.Publisher('/airgym/cmd', AttitudeTarget, queue_size=2000)
            self.action = torch.zeros(4, device=self.device)
        else:
            pass

        # 初始化频率相关变量
        self.last_time = time.time()
        self.message_count = 0
        self.frequency = 0

        # env settings
        self.has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        if has_masks_func:
            self.has_masks = self.env.has_action_mask()

        self.is_tensor_obses = True

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

        print("#" * 60)
        print("#" + " " * 15 + "Network has been LOADED !!!" + " " * 16 + "#")
        print("#" * 60)

        self.start_t = rospy.Time.now().to_sec()

    def depth_cb(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            rospy.loginfo("Depth Image Shape: {}".format(depth_image.shape))
            rospy.loginfo("Depth Image Type: {}".format(depth_image.dtype))
            
            center_x = depth_image.shape[1] // 2
            center_y = depth_image.shape[0] // 2
            center_depth = depth_image[center_y, center_x]
            rospy.loginfo("Center Pixel Depth: {:.2f} meters".format(center_depth / 1000.0))
            import matplotlib.pyplot as plt

            depth_clipped = np.clip(depth_image, 0, 6000)
            # print(F"np.max(depth_image): {np.max(depth_image)}")
            depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)

            depth_normalized = np.uint8(depth_normalized)

            depth_resized = cv2.resize(depth_normalized, (212, 120), interpolation=cv2.INTER_AREA)

            depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_VIRIDIS)
            colored_image = plt.cm.magma(depth_image / 6000.0)  

            colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  
            cv2.imshow(colored_image)
            cv2.waitKey(1)
 

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))


    def _callback(self, data):
        s = data.data
        self.target_state = torch.tensor(s, device=self.device)
        # print(self.target_state)
        # self.target_state = torch.tensor([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12]], 
        #                                  device=self.device)

    def callback(self, data):
        # Process the incoming message
        pose = data.pose.pose.position
        quat = data.pose.pose.orientation
        linvel = data.twist.twist.linear
        angvel = data.twist.twist.angular

        pose_tensor = torch.tensor([pose.x, pose.y, pose.z], device=self.device)
        quat_tensor = torch.tensor([quat.x, quat.y, quat.z, quat.w], device=self.device)
        linvel_tensor = torch.tensor([linvel.x, linvel.y, linvel.z], device=self.device)
        angvel_tensor = torch.tensor([angvel.x, angvel.y, angvel.z], device=self.device)

        root_matrix = quaternion_to_matrix(quat_tensor[[3, 0, 1, 2]]).flatten()
        assert root_matrix.shape[0] == 9

        ##------------------- normal obs ----------------------##
        # features = torch.rand(12, device=self.device)
        # real_obs = torch.cat((root_matrix, pose_tensor, linvel_tensor, angvel_tensor, self.action, features)).unsqueeze(0)
        # real_obs[:, :18] -= self.target_state

        ##------------------- depth obs ----------------------##
        # depth_image = torch.tensor([self.depth_image], device=self.device)
        

        ##------------------- tracking obs ----------------------##
        cur_t = rospy.Time.now().to_sec() - self.start_t
        n_steps = 10
        step_size = 5
        scale = 0.25 #0.25
        dt = 0.01
        seq_t = cur_t + torch.arange(n_steps, device=self.device) * step_size * dt
        seq_t = seq_t * scale

        ref_x = 3 * torch.sin(seq_t) / (1 + torch.cos(seq_t) ** 2)
        ref_y = 3 * torch.sin(seq_t) * torch.cos(seq_t) / (1 + torch.cos(seq_t) ** 2)
        ref_z = torch.ones_like(ref_x)
        traj = torch.stack((ref_x, ref_y, ref_z), dim=-1)
        related_pos = traj - pose_tensor
        related_pos = related_pos.flatten()

        # n_steps = 10
        # step_size = 5
        # scale = 1
        # dt = 0.01
        # seq_t = cur_t + torch.arange(n_steps, device=self.device) * step_size * dt
        # seq_t = seq_t * scale

        # ref_x = seq_t
        # ref_y = 1/(1+torch.pow(.5*seq_t-2, 2))-.25
        # ref_z = torch.ones_like(ref_x)
        # traj = torch.stack((ref_x, ref_y, ref_z), dim=-1)
        # related_pos = traj - pose_tensor
        # related_pos = related_pos.flatten()

        real_obs = torch.cat((root_matrix, pose_tensor, linvel_tensor, angvel_tensor, related_pos)).unsqueeze(0)


        ##--------------------- inference --------------------------------##
        action = self.inference(real_obs)
        self.action = action
        inference_res = Float64MultiArray()
        inference_res.data = action.cpu().numpy().tolist()
        self.action_pub.publish(inference_res)

        # Create the output message
        if self.ctl_mode == "pos":
            output_msg = PositionTarget()
            output_msg.position.x = action[0].cpu().numpy()
            output_msg.position.y = action[1].cpu().numpy()
            output_msg.position.z = action[2].cpu().numpy()
            output_msg.yaw = action[3].cpu().numpy()
            output_msg.type_mask = IGNORE_VX | IGNORE_VY | IGNORE_VZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | FORCE | IGNORE_YAW_RATE
        elif self.ctl_mode == "vel":
            output_msg = Twist()
            output_msg.linear.x = action[0].cpu().numpy()
            output_msg.linear.y = action[1].cpu().numpy()
            output_msg.linear.z = action[2].cpu().numpy()
        elif self.ctl_mode == "atti": # body_rate stores angular
            output_msg = AttitudeTarget()
            output_msg.orientation.x = action[1].cpu().numpy()
            output_msg.orientation.y = action[2].cpu().numpy()
            output_msg.orientation.z = action[3].cpu().numpy()
            output_msg.orientation.w = action[0].cpu().numpy()
            output_msg.thrust = action[4].cpu().numpy()
            output_msg.type_mask = IGNORE_ROLL_RATE | IGNORE_PITCH_RATE | IGNORE_YAW_RATE_
        elif self.ctl_mode == "rate":
            output_msg = AttitudeTarget()
            output_msg.body_rate.x = action[0].cpu().numpy()
            output_msg.body_rate.y = action[1].cpu().numpy()
            output_msg.body_rate.z = action[2].cpu().numpy()
            output_msg.thrust = action[3].cpu().numpy()*0.5 + 0.5
            output_msg.type_mask = IGNORE_ATTITUDE
        else:
            pass
        
        # Publish the message
        self.cmd_pub.publish(output_msg)

        # 更新频率检测
        self.message_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:  # 每秒计算一次频率
            self.frequency = self.message_count / (current_time - self.last_time)
            rospy.loginfo(f"CMD frequency: {self.frequency} Hz")
            self.message_count = 0
            self.last_time = current_time

    def inference(self, real_obses):
        if self.has_masks:
            masks = self.env.get_action_mask()
            action = self.get_masked_action(
                real_obses, masks, self.is_deterministic)
        else:
            action = self.get_action(real_obses, self.is_deterministic)
        
        if self.render_env:
            self.env.render(mode='human')
            time.sleep(self.render_sleep)
        return action
    
    def run(self):
        # get into loop
        while not rospy.is_shutdown():
            rospy.spin()
            self.rate.sleep()

    def env_step(self, env, actions):
        return super().env_step(env, actions)