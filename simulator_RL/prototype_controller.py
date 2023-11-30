from ast import If, Return
from collections import deque
import imp
from scipy.spatial.transform import Rotation as R
import casadi as ca
import numpy as np
import sys
import os
import time
from gait import Gait


class QP:
    def __init__(self):
        self.mu = 0.6
        self.u_min = 0.0
        self.u_max = 2000.0
        self.casadi_problem = self.gen_problem()

    def casadi_solve(self, A, b, Q, R, R_last, u_last, contact):
        try:
            u, cost = self.casadi_problem(A, b, Q, R, R_last, u_last, contact)
            return np.array(u.elements()).reshape(4, 3)
        except SystemError or RuntimeError:
            return np.zeros([4, 3])

    def gen_problem(self):
        opti = ca.Opti("conic")
        A = opti.parameter(6, 12)
        b = opti.parameter(6)
        Q = opti.parameter(6)
        R = opti.parameter(1)
        R_last = opti.parameter(1)
        contact = opti.parameter(4)
        u = opti.variable(12)
        u_last = opti.parameter(12)

        cost = ca.dot(Q * (A @ u - b), Q * (A @ u - b)) + R * \
            ca.dot(u, u) + R_last * ca.dot(u - u_last, u - u_last)
        opti.minimize(cost)
        for i in range(4):
            opti.subject_to(-u[3 * i + 2] <= self.u_min * contact[i])
            opti.subject_to(u[3 * i + 2] <= self.u_max * contact[i])
            opti.subject_to(u[3 * i + 0] <= self.mu * u[3 * i + 2])
            opti.subject_to(u[3 * i + 1] <= self.mu * u[3 * i + 2])
            opti.subject_to(-u[3 * i + 0] <= self.mu * u[3 * i + 2])
            opti.subject_to(-u[3 * i + 1] <= self.mu * u[3 * i + 2])

        option = dict()
        option["print_problem"] = False
        option["print_time"] = False
        option["printLevel"] = "none"
        option["error_on_fail"] = False

        opti.solver("qpoases", option)
        return opti.to_function("F", [A, b, Q, R, R_last, u_last, contact], [u, cost])


class QuadrupedController:
    def __init__(self):
        
        # gait
        self.use_event_based_gait = False
        self.use_payload_est=False
        self.gait = Gait()
        # counter
        self.counter = 0
        # quick stop
        self.quick_stop_ = False
        self.body_height = 0.64
        self.body_width = 0.44
        self.body_length = 0.8
        self.gait_period = 0.6
        delta_y = 0.00
        self.hip_position = np.array([[+self.body_length/2, -self.body_width/2+delta_y, 0],  # v1
                                      [+self.body_length/2,
                                       self.body_width/2-delta_y, 0],
                                      [-self.body_length/2, - \
                                          self.body_width/2+delta_y, 0],
                                      [-self.body_length/2, self.body_width/2-delta_y, 0]])
        self.root_pos_des_rel = np.zeros(3)
        self.root_pos_des_abs = np.zeros(3)
        self.root_acc_quick_stop = np.zeros(3)
        self.contact_target_last = np.zeros(4, dtype=bool)
        self.gait_changed_num = 0
        self.gait_type_last_stop = 1
        self.vel0 = np.zeros(2)
        self.pos0 = np.zeros(2)
        self.Tc = 0
        self.time = 0
        self.time_stop_total = 0
        self.quick_stop_first_run = True
        self.delta_xy = np.zeros(2)
        # test info
        self.root_acc_angle = np.zeros(3)
        # lip
        self.use_lip = False
        self.root_p0 = np.zeros(3)
        self.root_v0 = np.zeros(3)
        self.root_time = 0
        self.root_des = np.zeros(3)
        self.sim_step = 0.001
        self.total_mass = 0
        self.contact_est = [1]*4
        self.foot_force_body_cmd = np.zeros(12)

        # state
        self.root_pos = np.zeros(3)
        self.root_quat = np.zeros(4)
        self.root_vel_world = np.zeros(3)
        self.root_omega_world = np.zeros(3)
        self.root_acc_body = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.joint_tau = np.zeros(12)

        self.root_euler = np.zeros(3)
        self.use_terrain_est_new = True
        self.terrain_euler = np.zeros([3, 1])  # euler zyx [rz ry rx]
        self.rot_mat_terrain = np.zeros([3, 3])
        self.terrain_height = 0  # the projection point height of body com in terrain plane
        self.terrain_com_in_plane = np.zeros(3)
        self.terrain_coef = np.zeros([3, 1])
        self.rot_mat = np.zeros([3, 3])
        self.rot_mat_z = np.zeros([3, 3])

        self.root_lin_vel_rel = np.zeros(3)
        self.root_ang_vel_rel = np.zeros(3)
        # foot contact force from force sensor
        self.calf_contact_force = np.zeros([4, 3])
        self.foot_contact_force = np.zeros([4, 3])
        self.foot_pos_world = np.zeros([4, 3])
        self.foot_pos_abs = np.zeros([4, 3])
        self.foot_pos_rel = np.zeros([4, 3])
        self.foot_vel_world = np.zeros([4, 3])
        # estimated force calulated from jacobian and joint torque
        self.foot_force_est = np.zeros(12)
        self.foot_vel_abs = np.zeros([4, 3])
        self.foot_vel_rel = np.zeros([4, 3])
        self.foot_jaco = np.zeros([12, 12])
        self.jacob_world=np.zeros([12,18])

        self.contact_pos_world = np.zeros([4, 3])
        self.terrain_pitch = 0
        self.contact_pos_avg = np.zeros(3)
        self.kd_root_pos = np.array([0.3, 0.3, 0.0])
        self.gamepad_cmd = []

        # user
        self.gait_type = 0  # stand: 0, walk: 1, trot: 2, pace: 3
        self.gait_type_last = 0
        self.gait_stop_height = 0.10
        self.use_joy = True
        self.root_pos_target = np.zeros(3)
        self.root_euler_target = np.zeros(3)
        self.root_lin_vel_target = np.zeros(3)
        self.root_ang_vel_target = np.zeros(3)
        self.joy_value = np.zeros(8)

        # plan
        self.kp_foot_x = 0.0
        self.kd_foot_x = 0.0
        self.kf_foot_x = 0.0
        self.kp_foot_y = 0.0
        self.kd_foot_y = 0.0
        self.kf_foot_y = 0.0
        self.kp_pitch_z = 0.0
        global FOOT_ST
        global FOOT_STD
        global FOOT_SW
        FOOT_ST = 0
        FOOT_STD = 1
        FOOT_SW = 2

        self.foot_state = np.array([FOOT_ST, FOOT_ST, FOOT_ST, FOOT_ST])
        self.foot_to_sw = []
        self.foot_counter_st = np.zeros(4, dtype=float)
        self.foot_counter_sw = np.zeros(4, dtype=float)
        self.foot_counter_st_speed = np.zeros(4, dtype=float)
        self.foot_counter_sw_speed = np.zeros(4, dtype=float)

        self.contact_target = np.zeros(4, dtype=bool)
        self.foot_pos_world_target = np.zeros([4, 3])
        self.foot_pos_abs_target = np.zeros([4, 3])
        self.foot_pos_rel_target = np.zeros([4, 3])

        # command
        self.qp = QP()
        self.torque = np.zeros(12)

        self.default_foot_pos = np.array([[+self.body_length/2, -self.body_width/2, -self.body_height],  # v1
                                          [+self.body_length/2,
                                              self.body_width/2, -self.body_height],
                                          [-self.body_length/2, - \
                                              self.body_width/2, -self.body_height],
                                          [-self.body_length/2, self.body_width/2, -self.body_height]])
        self.foot_pos_start = self.default_foot_pos.copy()
        self.foot_pos_end = self.default_foot_pos.copy()
        self.kp_kin = np.array([[1000.0, 1000.0, 10000.0],
                                [1000.0, 1000.0, 10000.0],
                                [1000.0, 1000.0, 10000.0],
                                [1000.0, 1000.0, 10000.0]])
        self.default_root_state = np.array([0, 0, self.body_height, 0, 0, 0])
        self.com_offset = np.array([0, 0, 0])
        self.km_kin = np.array([0.1, 0.1, 0.02,
                                0.1, 0.1, 0.02,
                                0.1, 0.1, 0.02,
                                0.1, 0.1, 0.02])
        self.kp_root_lin = np.array([500.0, 500.0, 15000.0])
        self.kd_root_lin = np.array([500.0, 500.0, 500.0])
        self.kp_root_ang = np.array([200.0, 1000.0, 20.0])
        self.kd_root_ang = np.array([2.0, 2.0, 20.0])
        self.root_pos_delta_z = 0
        self.root_vel_delta_z = -0.5

        self.grf_last = np.zeros(12)  # initialize the last solve zeros

    def updateState(self, state):
        # update body data
        self.root_pos = state.trunk_pos
        self.root_quat = np.array(
            [state.imu_quat[1], state.imu_quat[2], state.imu_quat[3], state.imu_quat[0]])
        self.root_vel_world = state.trunk_vel_in_world
        self.root_omega_world = state.trunk_omega_in_world
        self.root_acc_body = state.imu_acc

        # calculate rotation matrix
        r = R.from_quat(self.root_quat)  # xyzw
        self.root_euler = r.as_euler('zyx', degrees=False)
        self.rot_mat = r.as_matrix()
        r = R.from_euler('z', self.root_euler[0])
        self.rot_mat_z = r.as_matrix()
        self.root_lin_vel_rel = self.root_vel_world @ self.rot_mat_z  # body to world
        self.root_ang_vel_rel = self.root_omega_world @ self.rot_mat
       # update joint data
        self.joint_pos = state.qpos
        self.joint_vel = state.qvel
        self.joint_tau = np.array(state.tau_applied)
        # update contact force and estimate contact state
        global contact_force_low
        contact_force_low = 20
        self.foot_contact_force = state.contact_force.copy()
        # estimate contact state
        for i in range(4):
            if np.linalg.norm(self.foot_contact_force[i]) >= contact_force_low:
                self.contact_est[i] = 1
            else:
                self.contact_est[i] = 0

        self.foot_pos_world = state.foot_pos_in_world.transpose()
        # self.foot_pos_world[:, 2] -= foot_radius
        self.foot_pos_abs = self.foot_pos_world - self.root_pos
        self.foot_pos_rel = self.foot_pos_abs @ self.rot_mat

        self.foot_jaco = state.jacob_body.copy()
        # print("state.jacob_body",state.jacob_body)
        self.jacob_world=state.jacob_world.copy()
        #update wbc_massmatrix wbc_corilios
        # update terrain
        global terrain_filter_rate
        terrain_filter_rate = 0.5
        self.terrainStateEst()
        p = self.terrain_euler[1]
        h = self.terrain_height
        self.terrain_pitch = (1 - terrain_filter_rate) * \
            self.terrain_pitch + terrain_filter_rate * p
        self.terrain_height = (1 - terrain_filter_rate) * \
            self.terrain_height + terrain_filter_rate * h

    def terrainStateEst(self):
        if self.counter <= 2:  # init contact_pos_world
            self.contact_pos_world = self.foot_pos_world.copy()
        for i in range(4):
            # if (self.contact_target[i] and np.linalg.norm(self.foot_contact_force[i]) > contact_force_low):
            if (self.gait.foot_state[i] == 1):
                self.contact_pos_world[i] = self.foot_pos_world[i]
        A = self.contact_pos_world.copy()
        # plane equation ax+by+cz=1
        coef = np.linalg.lstsq(A, np.ones(4), rcond=None)[0]
        ZERO = 0.00000001
        if np.abs(coef[2]) < ZERO and coef[2] < 0:
            coef[2] = -ZERO
        if np.abs(coef[2]) < ZERO and coef[2] > 0:
            coef[2] = ZERO
        self.terrain_coef = coef.copy()
        # calculate euler of terrain,
        # the x axis of terrain and the x axis of body are in a vertical plane
        v1 = np.array([1, 0, -coef[0]/coef[2]])
        v2 = np.array([0, 1, -coef[1]/coef[2]])
        normal = np.cross(v1, v2)
        nz = normal/np.linalg.norm(normal)
        nx_trunk = self.rot_mat[:, 0]
        ny = np.cross(nz, nx_trunk)
        ny = ny/np.linalg.norm(ny)
        nx = np.cross(ny, nz)
        nx = nx/np.linalg.norm(nx)
        self.rot_mat_terrain = np.vstack([nx, ny, nz]).T
        r = R.from_matrix(self.rot_mat_terrain)  # xyzw
        self.terrain_euler = r.as_euler('zyx', degrees=False)
        # print("terrain zyx",self.terrain_euler)
        # calcualte the terrain height, plane equation is ax+by+cz=1
        self.terrain_height = self.getPlanePointZ(
            coef, self.root_pos[0], self.root_pos[1])
        self.terrain_com_in_plane[0] = self.root_pos[0]
        self.terrain_com_in_plane[1] = self.root_pos[1]
        self.terrain_com_in_plane[2] = self.terrain_height
        # print("terrain_height",self.terrain_height)

    def getPlanePointZ(self, plane_coef, x, y):
        ZERO = 0.00000001
        a = plane_coef[0]
        b = plane_coef[1]
        c = plane_coef[2]
        assert (np.abs(c) > ZERO, "the norm of c must > ZERO!!")
        z = (1-a*x-b*y)/c
        return z

    def updateUser(self, cmd):
        self.gamepad_cmd = cmd
        if self.use_joy:
            self.root_lin_vel_target[0] = cmd.vel_cmd[0]
            self.root_lin_vel_target[1] = cmd.vel_cmd[1]
            self.root_ang_vel_target[2] = cmd.omega_cmd[2]
            if cmd.gait_type <= 4:
                if self.default_root_state[2] + cmd.body_height < self.gait_stop_height:
                    self.gait_type = 0
                    self.root_lin_vel_target[:] = 0
                    self.root_ang_vel_target[:] = 0
                else:
                    self.gait_type = cmd.gait_type
            else:
                vel_total = np.linalg.norm(
                    self.root_lin_vel_target) + np.linalg.norm(self.root_ang_vel_target)
                if vel_total < 1e-6:
                    self.gait_type = 0
                elif vel_total < 0.3:
                    self.gait_type = 1
                else:
                    self.gait_type = 2

        else:
            self.root_lin_vel_target[:] = [0.0, 0.0, 0.0]
            self.root_ang_vel_target[:] = [0.0, 0.0, 0.0]
            self.gait_type = 0

        self.root_pos_target = self.default_root_state[0:3] + self.com_offset + np.array([0.0, 0.0, self.gamepad_cmd.body_height])

        # print("terrain_euler",self.terrain_euler)
        # print("root_euler",self.root_euler)
        filter = 0.5
        nx_root = self.rot_mat_terrain[:, 0]
        ny_root = np.cross(np.array([0, 0, 1]), nx_root)
        ny_root = ny_root/np.linalg.norm(ny_root)  # unit
        nz_root = np.cross(nx_root, ny_root)
        rotm_des_root = np.vstack([nx_root, ny_root, nz_root]).T
        euler = rot2euler(rotm_des_root)
        # self.root_euler_target=self.root_euler_target*filter+(1-filter)*self.terrain_euler #zyx
        self.root_euler_target = self.root_euler_target*filter +(1-filter)*euler  # keep body's y axis horizental
        self.root_euler_target[0] += self.root_ang_vel_target[2]*self.sim_step
        self.root_pos_target += self.terrain_com_in_plane

    def resetController(self):
        self.root_euler_target[:] = self.default_root_state[3:6]

    def updatePlan(self):
        if self.use_event_based_gait:
            self.gait.updateEventBasedGait(self.contact_est)
        else:
            self.gait.updateTimeBaseGait()
        # foot plan
        self.kp_foot_x = 0.1
        self.kd_foot_x = 0.1
        # self.kf_foot_x = 0.3 if self.gait_type in [0, 1] else 0.05
        self.kf_foot_x = 0.05
        self.kp_foot_y = 0.1
        self.kd_foot_y = 0.1
        # self.kf_foot_y = 0.3 if self.gait_type in [0, 1] else 0.1
        self.kf_foot_y = 0.05
        self.kp_pitch_z = 0.1
        self.kp_foot_x = 0.0
        self.kp_foot_y = 0.0
        self.kd_foot_x = 0.1
        self.kd_foot_y = 0.1
        delta_foot_x = self.kp_foot_x * (self.root_pos[0] - self.root_pos_target[0]) + \
            self.kd_foot_x * (self.root_lin_vel_rel[0] - self.root_lin_vel_target[0]) + \
            self.kf_foot_x * self.root_lin_vel_rel[0]
        delta_foot_y = self.kp_foot_y * (self.root_pos[1] - self.root_pos_target[1]) + \
            self.kd_foot_y * (self.root_lin_vel_rel[1] - self.root_lin_vel_target[1]) + \
            self.kf_foot_y * self.root_lin_vel_rel[1]
        # TODO: add height change based on terrain roll
        for leg in range(4):
            hip_position_world = self.root_pos +self.rot_mat@self.hip_position[leg]
            delta_foot_rel = np.array([delta_foot_x, delta_foot_y, 0])
            delta_foot_world = self.rot_mat@delta_foot_rel
            foot_position_world = hip_position_world+delta_foot_world
            foot_height = self.getPlanePointZ(self.terrain_coef, foot_position_world[0], foot_position_world[1])-0.02
            foot_position_world[2] = foot_height
            self.foot_pos_world_target[leg] = foot_position_world.copy()
            self.foot_pos_abs_target[leg] = self.foot_pos_world_target[leg] - self.root_pos


    def updateCommand(self):
        # foot control
        foot_pos_final = self.foot_pos_abs_target @ self.rot_mat_z
        foot_pos_cur = self.foot_pos_abs @ self.rot_mat_z
        # foot_pos_cur = self.foot_pos_rel

        # TODO: clean up the following two if statements
        if self.counter == 0:
            self.foot_pos_start = foot_pos_cur

        bezier_time = np.zeros(4)

        for i in range(4):
            if self.gait.foot_state[i] == 1:
                self.foot_pos_start[i] = foot_pos_cur[i]
            else:
                bezier_time[i] = self.gait.foot_swing_phase[i]
                self.foot_pos_end[i] = foot_pos_final[i]
        foot_pos_target,foot_vel_target = self._get_bezier_curve(self.foot_pos_start, foot_pos_final, bezier_time)
        # foot_pos_target = foot_pos_final
        if(foot_pos_target[0,1]>-0.1):
            a=1
        # print("foot_pos_target",foot_pos_target)

        foot_pos_error = foot_pos_target - foot_pos_cur
        foot_force_kin = self.kp_kin * foot_pos_error
        foot_force_kin=foot_force_kin@self.rot_mat_z.T
        foot_force_kin=foot_force_kin@self.rot_mat
        foot_force_kin_flat = foot_force_kin.flatten()

        #update acc for wbc
        foot_vel_des_body=foot_vel_target@self.rot_mat_z.T@self.rot_mat
        foot_vel_des_flat=foot_vel_des_body.flatten()
        j_vel_des=np.linalg.lstsq(self.foot_jaco, foot_vel_des_flat, rcond=-1)[0]
        j_vel=self.joint_vel
        foot_p_err_body=foot_pos_error@self.rot_mat_z.T@self.rot_mat
        j_pos_des=np.zeros(12)
        for leg in range(4):
            delta_q=np.linalg.lstsq(self.foot_jaco[3*leg:3*leg+3,3*leg:3*leg+3], foot_p_err_body[leg], rcond=-1)[0]
            j_pos_des[3*leg:3*leg+3]=self.joint_pos[3*leg:3*leg+3]+delta_q
        j_pos=self.joint_pos
        j_kd=np.array([1,2,3,1,2,3,1,2,3,1,2,3])
        j_kp=np.array([1,2,3,1,2,3,1,2,3,1,2,3])*100

        # avoid straight knee singularity
        A_aug = np.vstack((self.foot_jaco, 0.1 * np.eye(12)))
        b_aug = np.hstack((self.km_kin * foot_force_kin_flat, np.zeros(12)))
        torque_kin = np.linalg.lstsq(A_aug, b_aug, rcond=-1)[0]

        # touch down control
        foot_pos_error = self.foot_pos_end - foot_pos_cur
        foot_force_down = self.kp_kin * foot_pos_error
        contact_force_high = 100
        foot_force_down[:, 2] = -contact_force_high
        foot_force_down_flat = foot_force_down.flatten()
        torque_down = np.linalg.solve(self.foot_jaco, self.km_kin * foot_force_down_flat)

        # swing foot reaches limit
        # if np.any(self.foot_counter_sw >= count_per_phase):
        #     self.root_pos_delta_z += self.root_vel_delta_z * self.sim_step
        # else:
        #     self.root_pos_delta_z *= (1 - self.sim_step)
        if np.any(self.gait.touchdown_late == 1):  # ???
            self.root_pos_delta_z += self.root_vel_delta_z * self.sim_step
        else:
            self.root_pos_delta_z *= (1 - self.sim_step)
        self.root_pos_target[2] += self.root_pos_delta_z

        # grf control
        foot_force_grf = -self._root_control()  # foot force in world frame
        # print("foot_force",foot_force_grf)
        foot_force_grf_rel = foot_force_grf @ self.rot_mat  # transfer to body frame
        foot_force_grf_rel_flat = foot_force_grf_rel.flatten()
        torque_grf = self.foot_jaco.T @ foot_force_grf_rel_flat
       
        # merge torque
        for i in range(4):
            if self.gait.touchdown_early[i]:
                a = 1
            # if self.contact_target[i]:
            if self.gait.foot_state[i]:
                self.torque[3 * i:3 * i + 3] = torque_grf[3 * i:3 * i + 3]
            else:
                # if self.foot_counter_sw[i] < count_per_phase:
                if self.gait.touchdown_late[i] == 0:
                    self.torque[3 * i:3 * i + 3] = torque_kin[3 * i:3 * i + 3]
                else:
                    self.torque[3 * i:3 * i + 3] = torque_down[3 * i:3 * i + 3]

        self.torque += np.array([-2.0, 0, 0, 2.0, 0, 0, -2.0, 0, 0, 2.0, 0, 0])
        self.foot_force_body_cmd = np.linalg.solve(self.foot_jaco.T, self.torque)
        # hip gravity compensation
        
    def updateCounter(self):
        self.counter += 1
        for i in range(4):
            if self.foot_state[i] == FOOT_ST:
                self.foot_counter_st[i] += 1  # self.foot_counter_st_speed[i]
            elif self.foot_state[i] == FOOT_SW:
                self.foot_counter_sw[i] += 1  # self.foot_counter_sw_speed[i]

    def _root_control(self):
        # TODO: fix the 180 degree bug
        # euler_error = self.root_euler_target - self.root_euler
        root_acc_target = np.zeros(6)
        # control angle
        r = R.from_euler('zyx', self.root_euler_target)
        R_des = r.as_matrix()
        R_scr = self.rot_mat.copy()
        R_err = R_scr.T@R_des
        axis, angle_error = rot2axisangle(R_err)
        root_acc_target[3:6] += self.kp_root_ang * axis*angle_error
        self.root_acc_angle = self.kp_root_ang * axis*angle_error

        root_acc_target[0:3] += self.kp_root_lin * \
            (self.root_pos_target - self.root_pos)
        root_acc_target[0:3] += (self.kd_root_lin * (self.root_lin_vel_target - self.root_lin_vel_rel)) @ \
            self.rot_mat_z.T

        root_acc_target[3:6] += self.kd_root_ang * \
            (self.root_ang_vel_target - self.root_ang_vel_rel)
        #update for wbc

        gravity = 9.81 * self.total_mass
        root_acc_target[2] += gravity

        inv_inertia_mat = np.zeros([6, 12])
        inv_inertia_mat[0:3, :] = np.tile(np.eye(3), 4)
        for i in range(4):
            # inv_inertia_mat[3:6, i * 3: i * 3 + 3] = self.rot_mat_z.T @ skew(self.foot_pos_abs[i, :] - self.com_offset)
            inv_inertia_mat[3:6, i * 3: i * 3 + 3] = self.rot_mat.T@skew(
                self.foot_pos_abs[i, :] - self.rot_mat @ self.com_offset)
        acc_weight = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        grf_weight = 1e-3
        grf_diff_weight = 1e-2
        grf = self.qp.casadi_solve(inv_inertia_mat,
                                   root_acc_target,
                                   acc_weight,
                                   grf_weight,
                                   grf_diff_weight,
                                   self.grf_last,
                                   self.gait.foot_state)
        self.grf_last = grf.flatten()  # Store this solve to use in the next solve
        return grf

    def _get_bezier_curve(self, foot_pos_start, foot_pos_final, bezier_time):
        foot_pos_target = np.zeros([4, 3])
        foot_vec_target = np.zeros([4, 3])
        for i in range(4):
            bezier_x = np.array([foot_pos_start[i, 0],
                                 foot_pos_start[i, 0],
                                 foot_pos_final[i, 0],
                                 foot_pos_final[i, 0],
                                 foot_pos_final[i, 0]])
            foot_pos_target[i, 0],foot_vec_target[i,0] = bezier_curve(bezier_time[i], bezier_x)

        for i in range(4):
            bezier_y = np.array([foot_pos_start[i, 1],
                                 foot_pos_start[i, 1],
                                 foot_pos_final[i, 1],
                                 foot_pos_final[i, 1],
                                 foot_pos_final[i, 1]])
            foot_pos_target[i, 1],foot_vec_target[i,1] = bezier_curve(bezier_time[i], bezier_y)

        for i in range(4):
            bezier_z = np.array([foot_pos_start[i, 2],
                                 foot_pos_start[i, 2],
                                 foot_pos_final[i, 2],
                                 foot_pos_final[i, 2],
                                 foot_pos_final[i, 2]])
            bezier_z[1] += 0.0
            bezier_z[2] += np.minimum(0.6, 3 *
                                      (self.default_root_state[2] - self.gait_stop_height))
            foot_pos_target[i, 2],foot_vec_target[i,2] = bezier_curve(bezier_time[i], bezier_z)

        return foot_pos_target,foot_vec_target

    def setTotalBodyMass(self, total_mass):
        self.total_mass = total_mass


def rot2axisangle(rotm):
    ZERO = 0.000000001
    r = R.from_matrix(rotm)
    rot_vec = r.as_rotvec()
    norm = np.linalg.norm(rot_vec)
    if norm < ZERO:
        axis = np.array([0, 0, 1])
        angle = 0
    else:
        axis = rot_vec/norm
        angle = norm
    return axis, angle


def rot2euler(rotm):
    r = R.from_matrix(rotm)
    euler = r.as_euler("zyx", degrees=False)
    return euler


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def bezier_curve(alpha, param):
    # 4th order bezier, 5 coefficient
    degree = 4
    coefficient = [1, 4, 6, 4, 1]
    y = 0
    dy=0
    for i in range(degree + 1):
        y += coefficient[i] * np.power(alpha, i) * np.power(1 - alpha, degree - i) * param[i]
        if i==0:
            dy+=-coefficient[i] *degree*np.power(1 - alpha, degree - 1) * param[i]
        elif i==degree:
            dy += coefficient[i] *i* np.power(alpha, i-1)* param[i]
        else:
            dy+=coefficient[i] *( i*np.power(alpha, i-1) * np.power(1 - alpha, degree - i)
            -np.power(alpha, i) * (degree - i)*np.power(1 - alpha, degree - i-1) )* param[i]
    return y,dy

