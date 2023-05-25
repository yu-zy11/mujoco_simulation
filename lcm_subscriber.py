import lcm
from scipy.spatial.transform import Rotation as R
# ******************** import lcm message files here
from lcm_package.MujocoCommand import MujocoCommand
from lcm_package.MujocoState import MujocoState
import threading
import time


class LcmInterface():
    def __init__(self):
        self.lc = lcm.LCM()
        # s***************et variables for lcm message data
        self.thread_id = threading.Thread(target=self.receiveMessageThread)
        self.subscribeLCM()
        self.cmd = MujocoCommand()
        self.state = MujocoState()
        self.mj_qpos = [0]*19
        self.isCommandReceived = False
    # ************** subscribe LCM data

    def subscribeLCM(self):
        subscription = self.lc.subscribe(
            "mujoco_cmd", self.MujocoCommand_callback)
        subscription.set_queue_capacity(1)  # store only one data

        self.thread_id.start()

    def MujocoCommand_callback(self, channel, data):
        self.cmd = MujocoCommand.decode(data)
        self.isCommandReceived = True
    # ***************set data to be published

    def publishLCM(self):
        self.lc.publish("mujoco_state", self.state.encode())

    # def getJointState(self):
    #     rpy = self.global_to_robot.rpy  # x  y z
    #     # scipy euler in z y x sequence
    #     r = R.from_euler('zyx', [rpy[2], rpy[1], rpy[0]], degrees=False)
    #     quat = r.as_quat()  # quat sequence is x y z w for scipy
    #     self.mj_qpos[0:3] = self.global_to_robot.xyz
    #     # quat sequence is w x y z  for mujoco
    #     self.mj_qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
    #     self.mj_qpos[7:19] = self.leg_control_data.q
    #     return self.mj_qpos

    def getRobotCmd(self):
        return self.cmd

    def receiveMessageThread(self):
        while (True):
            self.lc.handle()
