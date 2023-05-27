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
from quadruped_controller import QuadrupedController


class State:

    def __init__(self):
        #data from encoder and imu.default joint seq is FR:0-2 FL:3-5 RR:6-8 RL:9-11
        self.qpos = [0] * 12
        self.qvel = [0] * 12
        self.imu_acc = [0] * 3
        self.imu_omega = [0] * 3
        self.imu_quat = [0] * 3
        self.tau_applied = [0] * 12
        #data from state estimator,here we use cheater state
        self.trunk_pos = np.zeros((3, 1))
        self.trunk_vel_in_world = np.zeros((3, 1))
        self.trunk_vel_in_body = np.zeros((3, 1))
        self.rpy = np.zeros((3, 1))  #euler [z y x]
        self.rotR = np.eye(3)
        self.trunk_omega_in_world = np.zeros((3, 1))
        self.trunk_omega_in_body = np.zeros((3, 1))
        self.foot_pos_in_body = np.zeros(
            (3, 4))  #foot position relate to body com in body frame
        self.foot_pos_in_world = np.zeros(
            (3, 4))  #foot absolute position in world frame
        self.jacob_world = np.zeros((12, 18))
        self.jacob_body=np.zeros((12,12))
        self.contact_force=np.zeros(4)
        


class Command:
    def __init__(self):
        #default joint seq is FR:0-2 FL:3-5 RR:6-8 RL:9-11
        self.qpos_des = [0] * 12
        self.qvel_des = [0] * 12
        self.kp = [100] * 12
        self.kd = [2] * 12
        self.tau_ff = [0] * 12
class GamepadComamnd:
    def __init__(self):
        self.vel_cmd=[0]*3
        self.omega_cmd=[0]*3
        self.body_height=0
        self.gait_type=0


class MujocoSimulator:
    def __init__(self, model_file) -> None:
        self.state = State()
        self.cmd = Command()
        self.gamepad_cmd=GamepadComamnd()
        self.xml_path = model_file
        hip = -0.732
        knee = 1.4
        self.default_joint_pos = [0, hip, knee, 0, hip, knee, 0, hip, knee, 0, hip, knee  ]
        self.print_camera_config = 1
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0
        self.ctrl=QuadrupedController()
        # ******************setting for ros
        self.ros_thread = threading.Thread(target=self.publish2ros)
        rospy.init_node("mujoco_message", anonymous=True)
        self.jStatePub = JointInfoPub("joint_state")
        self.jCommandPub = JointInfoPub("joint_cmd")
        self.bdInfo = BodyInfoPub("body_cheater_state")

    def publish2ros(self):
        while (True):
            begin = time.time()
            self.getMujocoState()
            # publish joint command to ros
            mujoco_time = self.data.time
            self.jCommandPub.appendData(self.cmd.qpos_des, self.cmd.qvel_des,
                                        self.cmd.tau_ff, mujoco_time)
            self.jCommandPub.publishData()
            # publsih joint state to ros
            self.jStatePub.appendData(self.state.qpos, self.state.qvel,
                                      self.state.tau_applied, mujoco_time)
            self.jStatePub.publishData()
            # publish body cheater state
            q = [0] * 6
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
            while (end - begin < 0.001):
                time.sleep(0.0001)
                end = time.time()

    def resetSim(self):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[0:7]=[0, 0, 0.5, 1, 0, 0, 0]
        self.data.qpos[7:19]=self.default_joint_pos
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

    def setTorqueServo(
            self, actuator_no):  # set motor's control mode to be torque mode
        self.model.actuator_gainprm[actuator_no, 0:3] = [1, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, 0]
        self.model.actuator_biastype[actuator_no] = 0
        # print(self.model.actuator_biastype)

    def initSimulator(self):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)  # MuJoCo data
        self.cam = mj.MjvCamera()  # Abstract camera
        self.opt = mj.MjvOption()  # visualization options
        self.opt.flags[12] = 1  # show perturbation force
        self.opt.flags[14] = 1  # show contact area
        self.opt.flags[15] = 1  # show contact force
        # set camera configuration
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 2
        self.cam.lookat = np.array([0.0, 0.0, 0])
        self.data.qpos[7:19] = self.default_joint_pos.copy()
        # Init GLFW library, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1000, 900, "prototype_simulator",None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model,mj.mjtFontScale.mjFONTSCALE_150.value)
        # print camera configuration (help to initialize the view)
        if (self.print_camera_config == 1):
            print('cam.azimuth =', self.cam.azimuth, ';', 'cam.elevation =',
                  self.cam.elevation, ';', 'cam.distance = ',
                  self.cam.distance)
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
    def initController(self):
        total_mass=sum(self.model.body_mass[1:16])
        self.ctrl.setTotalBodyMass(total_mass)
        print("robot total mass is ",total_mass)

    def updateMujocoCmd(self):
        # self.cmd.qpos_des = self.default_joint_pos.copy()
        self.cmd.qpos_des = self.state.qpos.copy()
        self.cmd.tau_ff=self.ctrl.torque.tolist()
        self.cmd.kp=[0]*12
        self.cmd.kd=[2]*12
        a=1

    def controller(self, model, data):
        self.getMujocoState()
        self.ctrl.updateCounter()
        self.ctrl.updateState(self.state)
        # robot_state.update_output()
        self.ctrl.updateUser(self.gamepad_cmd)
        self.ctrl.updatePlan()
        self.ctrl.updateCommand()
        # robot_state.check_termination()
        self.updateMujocoCmd()
        cmd = self.cmd
        for i in range(12):
            self.data.ctrl[i] = cmd.kp[i] * (
                cmd.qpos_des[i] - self.data.qpos[7 + i]) + cmd.kd[i] * (
                    cmd.qvel_des[i] - self.data.qvel[6 + i]) + cmd.tau_ff[i]
        print("ctrl",self.data.ctrl)
        # time.sleep(0.0001)
    def runSimulation(self):
        self.ros_thread.start()
        # key and mouse control
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        mj.set_mjcb_control(self.controller)
        # mj.mj_forward(self.model, self.data)
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time-time_prev<1.0/60.0:
                # self.controller(self.model,self.data)
                mj.mj_step(self.model, self.data)

            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # Update scene and render
            self.cam.lookat = np.array(self.data.qpos[0:3])
            self.opt.flags[14] = 1  # show contact area
            self.opt.flags[15] = 1  # show contact force
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
            # while (self.data.time-time_prev < 1.0/60.0):
            #     print(self.data.time-time_prev)
        glfw.terminate()

    def getMujocoState(self):
        # imu
        self.state.imu_acc = self.data.sensordata[7:10]  # include gravity 9.81
        self.state.imu_omega = self.data.sensordata[4:7]
        self.state.imu_quat = self.data.sensordata[0:4]  #w x y z
        # encoder
        self.state.qpos = self.data.qpos[7:19]
        self.state.qvel = self.data.qvel[6:18]
        self.state.tau_applied = self.data.qfrc_actuator[6:18]
        # parameters for cheater mode
        quat = self.data.qpos[3:7]  # w x y z
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rotR = r.as_matrix()
        self.state.rotR = rotR
        self.state.trunk_pos = np.array(self.data.qpos[0:3])
        self.state.trunk_vel_in_world = np.array(self.data.qvel[0:3])
        self.state.trunk_vel_in_body = rotR.transpose(
        ) @ self.state.trunk_vel_in_world
        self.state.trunk_omega_in_body = np.array(self.state.imu_omega)
        self.state.trunk_omega_in_world = rotR@self.state.trunk_omega_in_body
        #foot position in body frame
        for i in range(4):
            self.state.foot_pos_in_body[:, i] = np.array(
                self.data.sensordata[10 + 3 * i:13 + 3 * i])
            self.state.foot_pos_in_world[:,i] = rotR @ self.state.foot_pos_in_body[:,i] + self.state.trunk_pos
        jacp = np.zeros((3, self.model.nv))
        rotR12_T=np.zeros((12,12))
        for i in range(4):
            mj.mj_jacSite(self.model, self.data, jacp, None,i + 1)  #i=0 means imu site
            self.state.jacob_world[3 * i:3 + 3 * i, :] = jacp.copy()
            rotR12_T[3*i:3*i+3,3*i:3*i+3]=rotR.T
        self.state.jacob_body=rotR12_T@self.state.jacob_world[:,6:19]
        # print(jacp)
        #contact force
        self.state.contact_force=np.array(self.data.sensordata[22:26])
        # void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]);

    def keyboard(self, window, key, scancode, act, mods):
        if (act == glfw.PRESS and key == glfw.KEY_R):
            self.resetSim()
            self.gamepad_cmd.gait_type=0
        if act == glfw.PRESS and key == glfw.KEY_UP:
            self.gamepad_cmd.vel_cmd[0]+=0.1
        if act == glfw.PRESS and key == glfw.KEY_DOWN:
            self.gamepad_cmd.vel_cmd[0]-=0.1
        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            self.gamepad_cmd.omega_cmd[2]+=0.1
        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            self.gamepad_cmd.omega_cmd[2]-=0.1
        if act==glfw.PRESS and key==glfw.KEY_W:
            self.gamepad_cmd.body_height+=0.05
        if act==glfw.PRESS and key==glfw.KEY_S:
            self.gamepad_cmd.body_height-=0.05
        if act==glfw.PRESS and key==glfw.KEY_0:
            self.gamepad_cmd.gait_type=0
        if act==glfw.PRESS and key==glfw.KEY_1:
            self.gamepad_cmd.gait_type=1
        if act==glfw.PRESS and key==glfw.KEY_2:
            self.gamepad_cmd.gait_type=2
        if act==glfw.PRESS and key==glfw.KEY_3:
            self.gamepad_cmd.gait_type=3
        # if act==glfw.PRESS and key==glfw.KEY_4:
        #     self.gamepad_cmd.gait_type=4

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
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene,
                          self.cam)

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
        mj.mjv_moveCamera(self.model, action, dx / height, dy / height,
                          self.scene, self.cam)


if __name__ == '__main__':
    model_xml = "prototype_model/scene_v1.xml"
    sim = MujocoSimulator(model_xml)
    sim.initSimulator()
    sim.initController()
    sim.runSimulation()
