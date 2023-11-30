import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time
import math
import threading
from scipy.spatial.transform import Rotation as R
from prototype_controller import QuadrupedController


class State:

    def __init__(self):
        # data from encoder and imu.default joint seq is FR:0-2 FL:3-5 RR:6-8 RL:9-11
        self.qpos = [0] * 12  #4条腿12个关节位置：运动范围参考微信图片
        self.qvel = [0] * 12  #关节速度，绝对值小于10rad/s
        self.imu_acc = [0] * 3  #IMU加速度计：±16G
        self.imu_omega = [0] * 3 #IMU 角速度：±450°/s
        self.imu_quat = [0] * 4 # IMU 姿态-四元数形式：满足四元数要求即可 
        self.tau_applied = [0] * 12 #每个关节施加扭矩：±180Nm
        # data from state estimator,here we use cheater state
        self.trunk_pos = np.zeros((3, 1)) #机身位置，无约束
        self.trunk_vel_in_world = np.zeros((3, 1)) #机身世界坐标系下速度：无约束
        self.trunk_vel_in_body = np.zeros((3, 1)) #机身体坐标系下速度：无约束
        self.rpy = np.zeros((3, 1))  #机身姿态-欧拉角形式： euler [z y x]满足欧拉角内在约束即可
        self.rotR = np.eye(3) #不用考虑
        self.trunk_omega_in_world = np.zeros((3, 1))#机身世界坐标系下角速度：无约束
        self.trunk_omega_in_body = np.zeros((3, 1))#机身体坐标系下角速度：无约束
        self.foot_pos_in_body = np.zeros((3, 4))  # foot position relate to body com in body frame 向量模不超过0.8m
        self.foot_pos_in_world = np.zeros((3, 4))  # foot absolute position in world frame 向量模不超过0.8m
        self.jacob_world = np.zeros((12, 18)) #不用考虑
        self.jacob_body = np.zeros((12, 12))#不用考虑
        self.contact_force = np.zeros(4)#不用考虑
        self.mass_matrix=np.zeros((18,18))#不用考虑
        self.coriolis=np.zeros((18,1))#不用考虑


class Command:
    def __init__(self):
        # default joint seq is FR:0-2 FL:3-5 RR:6-8 RL:9-11
        self.qpos_des = [0] * 12
        self.qvel_des = [0] * 12
        self.kp = [100] * 12
        self.kd = [2] * 12
        self.tau_ff = [0] * 12


class GamepadComamnd:
    def __init__(self):
        self.vel_cmd = [0]*3
        self.omega_cmd = [0]*3
        self.body_height = 0
        self.gait_type = 0


class MujocoSimulator:
    def __init__(self, model_file) -> None:
        self.fix_base=False
        self.use_ros_publish=False
        self.state = State()
        self.cmd = Command()
        self.gamepad_cmd = GamepadComamnd()
        self.xml_path = model_file
        hip = -0.732
        knee = 1.1
        self.default_joint_pos = [0, hip, knee, 0,
                                  hip, knee, 0, hip, knee, 0, hip, knee]
        # self.default_joint_pos = [0, -hip, -knee, 0, -hip, -knee, 0, hip, knee, 0, hip, knee  ]#for px2
        self.print_camera_config = 1
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0
        self.ctrl = QuadrupedController()
        self.time_pre=0

    def resetSim(self):
        # mj.mj_resetData(self.model, self.data)
        if self.fix_base:
            self.data.qpos = self.default_joint_pos.copy()
        else:
            self.data.qpos[0:7] = [0, 0, 0.7, 1, 0, 0, 0]
            self.data.qpos[7:19] = self.default_joint_pos.copy()
        # self.ctrl.resetController()
        mj.mj_forward(self.model, self.data)
        mj.mj_step(self.model, self.data)

    def stepSim(self,isUseRLCommand):
        self.controller(self.model, self.data,isUseRLCommand)
        mj.mj_step(self.model, self.data)
        if self.data.time-self.time_pre>1.0/60.0:
            #update viewer
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # Update scene and render
            if self.fix_base:
                self.cam.lookat = np.array([0,0,1])
            else:
                self.cam.lookat = np.array(self.data.qpos[0:3])
            self.opt.flags[14] = 1  # show contact area
            self.opt.flags[15] = 1  # show contact force
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
            self.time_pre = self.data.time

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
# set motor's control mode to be torque mode
    def setTorqueServo(self, actuator_no):  
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
        if len(self.data.qpos)==12:
            self.fix_base=True
        else:
            self.fix_base=False
        print("use fixed base:", self.fix_base)
        if self.fix_base:
            self.data.qpos = self.default_joint_pos.copy()
        else:
            self.data.qpos[7:19] = self.default_joint_pos.copy()
        # Init GLFW library, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1000, 900, "prototype_simulator", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        # print camera configuration (help to initialize the view)
        if (self.print_camera_config == 1):
            print('cam.azimuth =', self.cam.azimuth, ';', 'cam.elevation =',
                  self.cam.elevation, ';', 'cam.distance = ',self.cam.distance)
            print('cam.lookat =np.array([', self.cam.lookat[0], ',',
                  self.cam.lookat[1], ',', self.cam.lookat[2], '])')
        for i in range(12):
            self.setTorqueServo(i)
            # self.setPostionServo(i, 400)
        # self.setVelocityServo(4, 10)
        # self.setTorqueServo(7)
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        mj.mj_forward(self.model, self.data)

    def initController(self):
        total_mass = sum(self.model.body_mass[1:16])
        self.ctrl.setTotalBodyMass(total_mass)
        print("robot total mass is ", total_mass)

    def updateMujocoCmd(self):
        # self.cmd.qpos_des = self.default_joint_pos.copy()
        self.cmd.qpos_des = self.state.qpos.copy()
        self.cmd.tau_ff = self.ctrl.torque.tolist()
        self.cmd.kp = [0]*12
        self.cmd.kd = [2]*12
        return self.cmd

    def controller(self, model, data,isUseRLCommand):
        self.getMujocoState()
        if not isUseRLCommand:
            self.ctrl.updateCounter()
            self.ctrl.updateState(self.state)
            self.ctrl.updateUser(self.gamepad_cmd)
            self.ctrl.updatePlan()
            self.ctrl.updateCommand()
            cmd=self.updateMujocoCmd()
            self.setCommand(cmd.kp,cmd.kd,cmd.qpos_des,cmd.qvel_des,cmd.tau_ff)
        # robot_state.check_termination()
        

        # time.sleep(0.0001)
    def setCommand(self,kp,kd,qpos_des,qvel_des,tau_ff):
         for i in range(12):
            if self.fix_base:
                self.data.ctrl[i] = kp[i] * (qpos_des[i] - self.data.qpos[i]) + kd[i] * (
                    qvel_des[i] - self.data.qvel[i]) +tau_ff[i]
            else:
                self.data.ctrl[i] = kp[i] * (qpos_des[i] - self.data.qpos[7 + i]) + kd[i] * (
                    qvel_des[i] - self.data.qvel[6 + i]) +tau_ff[i]
            # self.data.ctrl[i]=qpos_des[i]

    def runSimulation(self):
        # key and mouse control
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        # mj.set_mjcb_control(self.controller)
        mj.mj_forward(self.model, self.data)
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time-time_prev < 1.0/60.0:
                self.controller(self.model, self.data,False)
                mj.mj_step(self.model, self.data)
            
            #update viewer
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # Update scene and render
            if self.fix_base:
                self.cam.lookat = np.array([0,0,1])
            else:
                self.cam.lookat = np.array(self.data.qpos[0:3])
            self.opt.flags[14] = 1  # show contact area
            self.opt.flags[15] = 1  # show contact force
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
        glfw.terminate()

    def getMujocoState(self):
        # update imu data
        self.state.imu_acc = self.data.sensordata[7:10]  # include gravity 9.81
        self.state.imu_omega = self.data.sensordata[4:7]
        self.state.imu_quat = self.data.sensordata[0:4]  # w x y z

        # update body data and joint data
        rotR=np.eye(3)
        if self.fix_base:
            self.state.qpos = self.data.qpos.copy()
            self.state.qvel = self.data.qvel[0:12]
            self.state.tau_applied = self.data.qfrc_actuator[0:12]
            quat =[1,0,0,0]   # w x y z
            self.state.rotR = np.eye(3)
            self.state.trunk_pos = np.array([0,0,0.6])
            self.state.trunk_vel_in_world = np.array([0,0,0])
            self.state.trunk_vel_in_body = np.array([0,0,0])
            self.state.trunk_omega_in_body =  np.array([0,0,0])
            self.state.trunk_omega_in_world =  np.array([0,0,0])
        else:
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
            self.state.trunk_vel_in_body = rotR.transpose() @ self.state.trunk_vel_in_world
            self.state.trunk_omega_in_body = np.array(self.state.imu_omega)
            self.state.trunk_omega_in_world = rotR@self.state.trunk_omega_in_body
        # foot position in body frame
        for i in range(4):
            self.state.foot_pos_in_body[:,i] = np.array(self.data.sensordata[10 + 3 * i:13 + 3 * i])
            self.state.foot_pos_in_world[:,i] = rotR @ self.state.foot_pos_in_body[:, i] + self.state.trunk_pos
        jacp = np.zeros((3, self.model.nv))
        rotR12_T = np.eye(12)
        if self.fix_base:
            self.state.jacob_world=np.zeros([12,12])
        else:
            self.state.jacob_world=np.zeros([12,18])

        for i in range(4):
            mj.mj_jacSite(self.model, self.data, jacp,None, i + 1)  # i=0 means imu site
            if self.fix_base:
                self.state.jacob_world[3*i:3 + 3*i,:] = jacp.copy()    
            else:
                self.state.jacob_world[3*i:3 + 3*i,:] = jacp.copy()
            rotR12_T[3*i:3*i+3, 3*i:3*i+3] = rotR.T

        if self.fix_base:
            self.state.jacob_body = rotR12_T@self.state.jacob_world[:,0:12]
        else:
            self.state.jacob_body = rotR12_T@self.state.jacob_world[:,6:19]
        # print("joint pos:",self.data.qpos)
        # print("jacob_world",self.state.jacob_world)
        Ma = np.zeros((self.model.nv, self.model.nv))
        mj.mj_fullM(self.model,Ma,self.data.qM)
        self.state.mass_matrix=Ma.copy()
        frc_bias= np.array([self.data.qfrc_bias])
        self.state.coriolis=frc_bias.copy()
        # print(jacp)
        # contact force
        self.state.contact_force = np.array(self.data.sensordata[22:26])
    def getRobotState(self):
        self.getMujocoState()
        return self.state
        # void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]);

    def keyboard(self, window, key, scancode, act, mods):
        if (act == glfw.PRESS and key == glfw.KEY_R):
            self.resetSim()
            self.gamepad_cmd.gait_type = 0
            self.gamepad_cmd.body_height = 0
            self.gamepad_cmd.vel_cmd = [0]*3
            self.gamepad_cmd.omega_cmd = [0]*3
        if act == glfw.PRESS and key == glfw.KEY_UP:
            self.gamepad_cmd.vel_cmd[0] += 0.1
            self.gamepad_cmd.omega_cmd[2] = 0
        if act == glfw.PRESS and key == glfw.KEY_DOWN:
            self.gamepad_cmd.vel_cmd[0] -= 0.1
            self.gamepad_cmd.omega_cmd[2] = 0
        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            self.gamepad_cmd.omega_cmd[2] += 0.3
        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            self.gamepad_cmd.omega_cmd[2] -= 0.3
        if act == glfw.PRESS and key == glfw.KEY_W:
            self.gamepad_cmd.body_height += 0.02
        if act == glfw.PRESS and key == glfw.KEY_S:
            self.gamepad_cmd.body_height -= 0.02
        if act == glfw.PRESS and key == glfw.KEY_0:
            self.gamepad_cmd.gait_type = 0
            self.ctrl.gait.setGaitType(0)
        if act == glfw.PRESS and key == glfw.KEY_1:
            self.gamepad_cmd.gait_type = 1
            self.ctrl.gait.setGaitType(1)
        if act == glfw.PRESS and key == glfw.KEY_2:
            self.gamepad_cmd.gait_type = 2
            self.ctrl.gait.setGaitType(2)
        if act == glfw.PRESS and key == glfw.KEY_3:
            self.gamepad_cmd.gait_type = 3
            self.ctrl.gait.setGaitType(3)
        if act == glfw.PRESS and key == glfw.KEY_4:
            self.gamepad_cmd.gait_type = 4
            self.ctrl.gait.setGaitType(4)
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
    model_xml = "prototype_model/scene_105.xml"
    # model_xml = "ls_dog105/scene.xml"
    sim = MujocoSimulator(model_xml)
    sim.initSimulator()
    sim.initController()
    sim.runSimulation()
