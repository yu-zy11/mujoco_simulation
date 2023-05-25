import lcm
import rospy
from std_msgs.msg import Float32MultiArray
# ******************** import lcm message files here
from leg_control_command_lcmt import leg_control_command_lcmt
from leg_control_data_lcmt import leg_control_data_lcmt
from localization_lcmt import localization_lcmt
from state_estimator_lcmt import state_estimator_lcmt
from MujocoState import MujocoState

import time
import threading


class RosPublisher:
    def __init__(self, topic_name="robot_topic"):
        # rospy.init_node(topic_name, anonymous=True)
        self.pub = rospy.Publisher(
            topic_name, Float32MultiArray, queue_size=10)
        self.multi_data = Float32MultiArray()

    def appendData(self, data):
        self.multi_data.data.append(data)

    def clearData(self):
        self.multi_data.data.clear()

    def publishData(self):
        self.pub.publish(self.multi_data)
        self.clearData()


class LCM2ROS():
    def __init__(self):
        rospy.init_node("ros_lcm_node", anonymous=True)
        self.lc = lcm.LCM()
        # s***************et variables for lcm message data
        self.leg_control_command = leg_control_command_lcmt()
        self.leg_control_data = leg_control_data_lcmt()
        self.global_to_robot = localization_lcmt()
        self.state_estimate = state_estimator_lcmt()

        self.state_estimate_pub = RosPublisher("state_estimator")
        # *****************set rostopic for publish
        self.ros_leg_command = RosPublisher("leg_control_command")
        self.ros_body_state = RosPublisher("body_state")
        self.ros_leg_state = RosPublisher("leg_state")
        self.thread_id = threading.Thread(target=self.lcmHandleThread)

    # ************** subscribe LCM data

    def subscribeLCM(self):
        self.subscription1 = self.lc.subscribe(
            "leg_control_command", self.leg_control_command_callback)
        self.subscription1.set_queue_capacity(100)  # store only one data
        self.subscription3 = self.lc.subscribe(
            "leg_control_data", self.leg_control_data_callback)
        self.subscription3.set_queue_capacity(100)  # store only one data
        self.subscription4 = self.lc.subscribe(
            "global_to_robot", self.global_to_robot_callback)
        self.subscription4.set_queue_capacity(100)  # store only one data
        self.subscription4 = self.lc.subscribe(
            "state_estimator", self.state_esitmator_callback)
        self.subscription4.set_queue_capacity(100)  # store only one data
        self.thread_id.start()
    # ***************register callback function

    def leg_control_command_callback(self, channel, data):
        self.leg_control_command = leg_control_command_lcmt.decode(data)

    def leg_control_data_callback(self, channel, data):
        self.leg_control_data = leg_control_data_lcmt.decode(data)

    def global_to_robot_callback(self, channel, data):
        self.global_to_robot = localization_lcmt.decode(data)

    def state_esitmator_callback(self, channel, data):
        self.state_estimate = state_estimator_lcmt.decode(data)
    # ***************set data to be published

    def publish2ROS(self):
        # leg command
        for i in range(12):  # qdes:0-11
            self.ros_leg_command.appendData(self.leg_control_command.q_des[i])
        for i in range(12):  # dq_des:12-23
            self.ros_leg_command.appendData(self.leg_control_command.qd_des[i])
        for i in range(12):  # tau_ff:24-35
            self.ros_leg_command.appendData(self.leg_control_command.tau_ff[i])
        self.ros_leg_command.publishData()
        # body state
        for i in range(3):  # body xyz:0-2
            self.ros_body_state.appendData(self.global_to_robot.xyz[i])
        for i in range(3):  # body rpy:3-5
            self.ros_body_state.appendData(self.global_to_robot.rpy[i])
        for i in range(3):  # body vxyz:6-8
            self.ros_body_state.appendData(self.global_to_robot.vxyz[i])
        for i in range(3):  # body omega_rpy:9-11
            self.ros_body_state.appendData(self.global_to_robot.omegaBody[i])
        self.ros_body_state.publishData()
        # leg state
        for i in range(12):  # q:0-11
            self.ros_leg_state.appendData(self.leg_control_data.q[i])
        for i in range(12):  # qd:12-23
            self.ros_leg_state.appendData(self.leg_control_data.qd[i])
        for i in range(12):  # tau:24-35
            self.ros_leg_state.appendData(self.leg_control_data.tau_est[i])
        self.ros_leg_state.publishData()
        # state estimator

    def lcmHandleThread(self):
        while (True):
            # time.sleep(0.001)
            self.lc.handle()


if __name__ == '__main__':
    print("start lcm test ")
    sim = LCM2ROS()
    sim.subscribeLCM()  # 订阅
    #
    while True:
        # sim.lc.handle()     #更新
        # sim.lc.handle_timeout(0.1)
        begin = time.time()
        sim.publish2ROS()
        end = time.time()
        while (end-begin < 0.002):
            time.sleep(0.0001)
            end = time.time()
