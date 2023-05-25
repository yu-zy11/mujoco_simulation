import lcm
from lcm_package.MujocoCommand import MujocoCommand
from lcm_package.MujocoState import MujocoState
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time
import math
import threading
from ros_pub import JointInfoPub
from ros_pub import BodyInfoPub
import rospy
from scipy.spatial.transform import Rotation as R


class MujocoSimulator:
    def __init__(self, model_file) -> None:
        self.xml_path = model_file
        hip = -0.732
        knee = 1.4
        self.default_joint_pos = [0, hip, knee, 0,
                                  hip, knee, 0, hip, knee, 0, hip, knee]
        self.print_camera_config = 1
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0
        # self.lcm = LcmInterface()
        # ******************settings for lcm
        self.lc = lcm.LCM()
        self.command = MujocoCommand()
        self.state = MujocoState()
        sub = self.lc.subscribe("mujoco_cmd", self.lcmCallback)
        sub.set_queue_capacity(1)  # store only one data

        # self.initSimulator()
        self.lcm_thread = threading.Thread(target=self.lcmHandleThread)
        self.lcm_Publish_thread = threading.Thread(target=self.lcmPublish)
        self.ros_thread = threading.Thread(target=self.publish2ros)
        # ******************setting for ros
        rospy.init_node("mujoco_message", anonymous=True)
        self.jStatePub = JointInfoPub("joint_state")
        self.jCommandPub = JointInfoPub("joint_cmd")
        self.bdInfo = BodyInfoPub("body_cheater_state")

    def lcmCallback(self, channel, data):
        self.command = MujocoCommand.decode(data)
        self.simulationStep()

    def lcmPublish(self):
        while (True):
            begin = time.time()
            self.updateState()
            self.lc.publish("mujoco_state", self.state.encode())
            end = time.time()
            while (end-begin < 0.001):
                time.sleep(0.0001)
                end = time.time()

    def lcmHandleThread(self):
        while (True):
            self.lc.handle()

    def publish2ros(self):
        while (True):
            begin = time.time()
            self.updateState()
            # joint command
            q_des = [0]*12
            qd_des = [0]*12
            tau_ff = [0]*12
            for i in range(4):
                q_des[3*i] = self.command.q_des_abad[i]
                q_des[3*i+1] = self.command.q_des_hip[i]
                q_des[3*i+2] = self.command.q_des_knee[i]
                qd_des[3*i] = self.command.qd_des_abad[i]
                qd_des[3*i+1] = self.command.qd_des_hip[i]
                qd_des[3*i+2] = self.command.qd_des_knee[i]
                tau_ff[3*i] = self.command.tau_abad_ff[i]
                tau_ff[3*i+1] = self.command.tau_hip_ff[i]
                tau_ff[3*i+2] = self.command.tau_knee_ff[i]
            mujoco_time = self.data.time
            self.jCommandPub.appendData(q_des, qd_des, tau_ff, mujoco_time)
            self.jCommandPub.publishData()
            # joint state
            qpos = self.data.qpos[7:19]
            qvel = self.data.qvel[6:18]
            tau_actual = self.data.qfrc_actuator[6:18]
            self.jStatePub.appendData(qpos, qvel, tau_actual, mujoco_time)
            self.jStatePub.publishData()
            # body cheater state
            q = [0]*6
            q[0:3] = self.data.qpos[0:3]
            quat = self.data.qpos[3:7]  # w x y z
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            rpy = r.as_euler("zyx", degrees=False)  # z y x
            v = self.data.qvel[0:6]
            q[3:6] = [rpy[2], rpy[1], rpy[0]]
            self.bdInfo.appendData(q, v, mujoco_time)
            self.bdInfo.publishData()
            # sleep
            end = time.time()
            while (end-begin < 0.001):
                time.sleep(0.0001)
                end = time.time()

    def resetSim(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        mj.mj_step(self.model, self.data)

    # set motor's control mode to be position mode
    def setPostionServo(self, actuator_no, kp):
        self.model.actuator_gainprm[actuator_no, 0:3] = [kp, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, -kp, 0]
        self.model.actuator_biastype[actuator_no] = 1
        # print(self.model.actuator_biastype)

    # set motor's control mode to be velocity mode
    def setVelocityServo(self, actuator_no, kv):
        self.model.actuator_gainprm[actuator_no, 0:3] = [kv, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, -kv]
        self.model.actuator_biastype[actuator_no] = 1
        # print(self.model.actuator_biastype)

    def setTorqueServo(self, actuator_no):  # set motor's control mode to be torque mode
        self.model.actuator_gainprm[actuator_no, 0:3] = [1, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, 0]
        self.model.actuator_biastype[actuator_no] = 0
        # print(self.model.actuator_biastype)

    def initSimulator(self):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options
        self.opt.flags[14] = 1  # show contact area
        self.opt.flags[15] = 1  # show contact force
        # set camera configuration
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 2
        self.cam.lookat = np.array([0.0, 0.0, 0])
        # print(self.data.qpos)
        if (self.data.qpos.size > 12):  # float base model
            self.data.qpos[7:19] = self.default_joint_pos.copy()
        else:  # fixed base model
            self.data.qpos = self.default_joint_pos.copy()
        # Init GLFW library, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1000, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        # print camera configuration (help to initialize the view)
        if (self.print_camera_config == 1):
            print('cam.azimuth =', self.cam.azimuth, ';', 'cam.elevation =',
                  self.cam.elevation, ';', 'cam.distance = ', self.cam.distance)
            print('cam.lookat =np.array([', self.cam.lookat[0], ',',
                  self.cam.lookat[1], ',', self.cam.lookat[2], '])')
        for i in range(12):
            self.setTorqueServo(i)
        # self.setVelocityServo(4, 10)
        # self.setTorqueServo(7)
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        mj.mj_forward(self.model, self.data)

    # def controller(self, model, data):
    def controller(self):
        cmd = self.command
        for i in range(4):
            self.data.ctrl[3*i] = cmd.kp_abad[i]*(cmd.q_des_abad[i]-self.data.qpos[7+3*i])+cmd.kd_abad[i]*(
                cmd.qd_des_abad[i]-self.data.qvel[6+3*i])+cmd.tau_abad_ff[i]
            self.data.ctrl[3*i+1] = cmd.kp_hip[i]*(cmd.q_des_hip[i]-self.data.qpos[7+3*i+1])+cmd.kd_hip[i]*(
                cmd.qd_des_hip[i]-self.data.qvel[6+3*i+1])+cmd.tau_hip_ff[i]
            self.data.ctrl[3*i+2] = cmd.kp_knee[i]*(cmd.q_des_knee[i]-self.data.qpos[7+3*i+2])+cmd.kd_knee[i]*(
                cmd.qd_des_knee[i]-self.data.qvel[6+3*i+2])+cmd.tau_knee_ff[i]
        # print("force", self.data.qfrc_actuator[6:18]-self.data.ctrl[24:36])

    def simulationStep(self):
        self.controller()
        mj.mj_step(self.model, self.data)

    def runSimulation(self):
        self.lcm_thread.start()
        self.ros_thread.start()
        self.lcm_Publish_thread.start()
        # key and mouse control
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        # mj.mj_forward(self.model, self.data)
        while not glfw.window_should_close(self.window):
            # mj.mj_forward(self.model, self.data)
            time_prev = self.data.time
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        # Update scene and render
            self.cam.lookat = np.array(self.data.qpos[0:3])
            self.opt.flags[14] = 1  # show contact area
            self.opt.flags[15] = 1  # show contact force
            mj.mjv_updateScene(self.model, self.data, self.opt, None,
                               self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
        # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
        # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
            # while (self.data.time-time_prev < 1.0/60.0):
            #     print(self.data.time-time_prev)
        glfw.terminate()

    def updateState(self):
        # imu
        self.state.imu_acc = self.data.sensordata[7:10]  # include gravity 9.81
        self.state.imu_omega = self.data.sensordata[4:7]
        self.state.imu_quat = [
            self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2], self.data.sensordata[3]]
        # encoder
        self.state.q_abad = [
            self.data.qpos[7], self.data.qpos[10], self.data.qpos[13], self.data.qpos[16]]
        self.state.q_hip = [
            self.data.qpos[8], self.data.qpos[11], self.data.qpos[14], self.data.qpos[17]]
        self.state.q_knee = [
            self.data.qpos[9], self.data.qpos[12], self.data.qpos[15], self.data.qpos[18]]
        self.state.qd_abad = [
            self.data.qvel[6], self.data.qvel[9], self.data.qvel[12], self.data.qvel[15]]
        self.state.qd_hip = [
            self.data.qvel[7], self.data.qvel[10], self.data.qvel[13], self.data.qvel[16]]
        self.state.qd_knee = [
            self.data.qvel[8], self.data.qvel[11], self.data.qvel[14], self.data.qvel[17]]
        # torque applied
        tau = self.data.qfrc_actuator[6:18]
        self.state.tau_abad = [tau[0], tau[3], tau[6], tau[9]]
        self.state.tau_hip = [tau[1], tau[4], tau[7], tau[10]]
        self.state.tau_knee = [tau[2], tau[5], tau[8], tau[11]]
        # parameter for cheater mode
        self.state.body_position = [
            self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]]
        self.state.body_velocity = [
            self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]]

    def keyboard(self, window, key, scancode, act, mods):
        if (act == glfw.PRESS and key == glfw.KEY_R):
            self.resetSim()
        if act == glfw.PRESS and key == glfw.KEY_S:
            print('Pressed key s')
        if act == glfw.PRESS and key == glfw.KEY_UP:
            vel_z += 2
        if act == glfw.PRESS and key == glfw.KEY_DOWN:
            vel_z -= 2
        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            vel_x -= 2
        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            vel_x += 2
    # update button state

    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        glfw.get_cursor_pos(window)  # update mouse position

    def mouse_scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                          yoffset, self.scene, self.cam)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.last_mouse_posx
        dy = ypos - self.last_mouse_posy
        self.last_mouse_posx = xpos
        self.last_mouse_posy = ypos
        # # determine action based on mouse button
        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_middle:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        width, height = glfw.get_window_size(window)  # get current window size
        mj.mjv_moveCamera(self.model, action, dx/height,
                          dy/height, self.scene, self.cam)


# #get the full path
# dirname = os.path.dirname(__file__)
# abspath = os.path.join(dirname + "/" + xml_path)
# xml_path = abspath
if __name__ == '__main__':
    model_xml = "/home/yu/workspace/python_work/model/unitree_a1/scene.xml"
    model_xml = "/home/yu/workspace/python_work/model/eame3_mujoco/scene.xml"
    sim = MujocoSimulator(model_xml)
    sim.initSimulator()
    sim.runSimulation()
