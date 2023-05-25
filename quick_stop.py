from collections import deque
from isaacgym import gymapi, gymutil, gymtorch
from scipy.spatial.transform import Rotation as R
from visual_ros import VisualRos
import casadi as ca
import numpy as np
import sys, os
import torch
# import window_jumper
import time

np.set_printoptions(linewidth=1000, precision=3)
torch.set_printoptions(linewidth=1000, precision=3)


class QP:
    def __init__(self):
        self.mu = 0.6
        self.u_min = 0.0
        self.u_max = 500.0
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

        cost = ca.dot(Q * (A @ u - b), Q * (A @ u - b)) + R * ca.dot(u, u) + R_last * ca.dot(u - u_last, u - u_last)
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


class RobotState:
    def __init__(self):
        # counter
        self.counter = 0
        # quick stop
        self.quick_stop_=False
        self.body_height=0.46
        self.root_pos_des_rel=np.zeros(3)
        self.root_pos_des_abs=np.zeros(3)
        self.root_acc_quick_stop=np.zeros(3)
        self.contact_target_last = np.zeros(4, dtype=bool)
        self.gait_changed_num=0
        self.gait_type_last_stop=1
        self.vel0=np.zeros(2)
        self.pos0=np.zeros(2)
        self.Tc=0
        self.time=0
        self.lip_stop_duration=0
        self.quick_stop_first_run=True
        self.delta_xy=np.zeros(2)
        self.stop_distance=np.zeros(3)
        self.stop_distance0=np.zeros(3)
        self.quick_stop_duration=0
        # lip
        self.use_lip=True
        
        self.root_p0=np.zeros(3)
        self.root_v0=np.zeros(3)
        self.root_time=0
        self.root_des=np.zeros(3)

        # state
        self.root_pos = np.zeros(3)
        self.root_quat = np.zeros(4)
        self.root_lin_vel = np.zeros(3)
        self.root_ang_vel = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)

        self.root_euler = np.zeros(3)
        self.rot_mat = np.zeros([3, 3])
        self.rot_mat_z = np.zeros([3, 3])

        self.root_lin_vel_rel = np.zeros(3)
        self.root_ang_vel_rel = np.zeros(3)

        self.calf_contact_force = np.zeros([4, 3])
        self.foot_contact_force = np.zeros([4, 3])
        self.foot_pos_world = np.zeros([4, 3])
        self.foot_pos_abs = np.zeros([4, 3])
        self.foot_pos_rel = np.zeros([4, 3])
        self.foot_vel_world = np.zeros([4, 3])
        self.foot_vel_abs = np.zeros([4, 3])
        self.foot_vel_rel = np.zeros([4, 3])
        self.foot_jaco = np.zeros([12, 12])

        self.contact_pos_world = np.zeros([4, 3])
        self.terrain_pitch = 0
        self.terrain_height = 0
        self.contact_pos_avg = np.zeros(3)
        self.kd_root_pos = np.array([0.3, 0.3, 0.0]) * count_per_cycle / 120

        # user
        self.gait_type = 0  # stand: 0, walk: 1, trot: 2, pace: 3
        self.gait_type_last = 0
        self.gait_stop_height = 0.3
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
        self.foot_state = np.array([FOOT_ST, FOOT_ST, FOOT_ST, FOOT_ST])
        self.foot_to_sw = []
        self.foot_counter_st = np.zeros(4, dtype=float)
        self.foot_counter_sw = np.zeros(4, dtype=float)
        # self.foot_counter_st_speed = np.zeros(4, dtype=float)
        # self.foot_counter_sw_speed = np.zeros(4, dtype=float)

        self.contact_target = np.zeros(4, dtype=bool)
        self.foot_pos_world_target = np.zeros([4, 3])
        self.foot_pos_abs_target = np.zeros([4, 3])
        self.foot_pos_rel_target = np.zeros([4, 3])

        # command
        self.qp = QP()
        self.torque = np.zeros(12)
        self.foot_pos_start = default_foot_pos.copy()
        self.foot_pos_end = default_foot_pos.copy()
        self.kp_kin = np.array([[1000.0, 1000.0, 5000.0],
                                [1000.0, 1000.0, 5000.0],
                                [1000.0, 1000.0, 5000.0],
                                [1000.0, 1000.0, 5000.0]])
        self.km_kin = np.array([0.1, 0.1, 0.02,
                                0.1, 0.1, 0.02,
                                0.1, 0.1, 0.02,
                                0.1, 0.1, 0.02])
        self.kp_root_lin = np.array([0000.0, 0000.0, 10000.0])
        self.kd_root_lin = np.array([500.0, 500.0, 500.0])
        self.kp_root_ang = np.array([200.0, 200.0, 20.0])
        self.kd_root_ang = np.array([2.0, 2.0, 20.0])
        self.root_pos_delta_z = 0
        self.root_vel_delta_z = -0.5

        self.grf_last = np.zeros(12)  # initialize the last solve zeros

    def reset(self):
        self.__init__()
        root_state[0] = torch.from_numpy(init_root_state).to(torch.float)
        root_state[1] = torch.from_numpy(init_terrain_state).to(torch.float)
        dof_position[:] = torch.from_numpy(default_dof_position).to(torch.float)
        dof_velocity[:] = torch.from_numpy(default_dof_velocity).to(torch.float)

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

    def update_state(self):
        # update terrain
        global init_terrain_state
        init_terrain_state[4] = np.clip(visual_ros.user_input.data[8], -0.2, 0.2)
        root_state[1] = torch.from_numpy(init_terrain_state).to(torch.float)
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))

        # update robot
        self.root_pos = root_state[0, 0:3].numpy()
        self.root_quat = root_state[0, 3:7].numpy()
        self.root_lin_vel = root_state[0, 7:10].numpy()
        self.root_ang_vel = root_state[0, 10:13].numpy()
        self.joint_pos = dof_position.numpy()
        self.joint_vel = dof_velocity.numpy()

        r = R.from_quat(self.root_quat)
        self.root_euler = r.as_euler('xyz')
        self.rot_mat = r.as_matrix()
        r = R.from_euler('z', self.root_euler[2])
        self.rot_mat_z = r.as_matrix()

        self.root_lin_vel_rel = self.root_lin_vel @ self.rot_mat_z #body to world
        self.root_ang_vel_rel = self.root_ang_vel @ self.rot_mat

        self.calf_contact_force = net_contact_force[calf_index].numpy()
        self.foot_contact_force = net_contact_force[foot_index].numpy()
        self.foot_pos_world = rigid_body_state[foot_index, 0:3].numpy()
        self.foot_pos_world[:, 2] -= foot_radius
        self.foot_pos_abs = self.foot_pos_world - self.root_pos
        self.foot_pos_rel = self.foot_pos_abs @ self.rot_mat
        self.foot_vel_world = rigid_body_state[foot_index, 3:6].numpy()
        self.foot_vel_abs = self.foot_vel_world - self.root_lin_vel  # TODO: miss the angular velocity
        self.foot_vel_rel = self.foot_vel_abs @ self.rot_mat
        foot_jaco = jacobian[0, foot_index, 0:3, 6:].numpy()
        foot_jaco = np.repeat(self.rot_mat.T[None, ...], 4, axis=0) @ foot_jaco
        self.foot_jaco = foot_jaco.reshape(12, 12)

        # terrain estimation
        for i in range(4):
            if self.counter == 0 or \
                    (self.contact_target[i] and np.linalg.norm(self.foot_contact_force[i]) > contact_force_low):
                self.contact_pos_world[i] = self.foot_pos_world[i]
        contact_pos_abs = self.contact_pos_world @ self.rot_mat_z
        contact_pos_x = contact_pos_abs[:, 0]
        contact_pos_z = contact_pos_abs[:, 2]

        A = np.vstack([contact_pos_x, np.ones(len(contact_pos_x))]).T
        m, c = np.linalg.lstsq(A, contact_pos_z, rcond=None)[0]
        p = np.arctan(-m)
        w = [1, 1, 1, 1]
        h = np.average(contact_pos_z, weights=w)

        self.terrain_pitch = (1 - terrain_filter_rate) * self.terrain_pitch + terrain_filter_rate * p
        self.terrain_height = (1 - terrain_filter_rate) * self.terrain_height + terrain_filter_rate * h

    def update_output(self):
        joint_torque = self.torque - self.joint_vel * dof_props["damping"]

        for i in range(12):  # 0-11
            visual_ros.append_data(self.joint_pos[i])
        for i in range(12):  # 12-23
            visual_ros.append_data(self.joint_vel[i])
        for i in range(12):  # 24-35
            visual_ros.append_data(joint_torque[i])
        for i in range(3):  # 36-38
            visual_ros.append_data(self.root_lin_vel[i])
        for i in range(3):  # 39-41
            visual_ros.append_data(self.root_pos[i])
        for i in range(12):  # 42-53
            visual_ros.append_data(self.grf_last[i])
        
        quick_stop_ros.append_data(self.quick_stop_) #0
        quick_stop_ros.append_data(self.gait_type) #1
        for i in range(3):#2-4
                quick_stop_ros.append_data(self.root_pos_target[i]-self.root_des[i]) 
        for i in range(3):#5-7
                quick_stop_ros.append_data(self.root_pos[i]-self.root_des[i])            
        for i in range(3):#8-10
            quick_stop_ros.append_data(self.root_lin_vel_target[i])
        for i in range(3):#11-13
            quick_stop_ros.append_data(self.root_acc_quick_stop[i])
        quick_stop_ros.append_data(self.quick_stop_duration) #14
        for i in range(3):#15 -17
            quick_stop_ros.append_data(self.stop_distance[i]) 
        quick_stop_ros.append_data(self.joy_value[7])#18
        for i in range(3):#self.foot_pos_rel_target
             quick_stop_ros.append_data(self.foot_pos_rel_target[i,0])
             quick_stop_ros.append_data(self.foot_pos_rel_target[i,1])
             quick_stop_ros.append_data(self.foot_pos_rel_target[i,2])


    def update_user(self):
        if self.use_joy:
            self.root_lin_vel_target[0:2] = self.joy_value[0:2]
            self.root_ang_vel_target[:] = self.joy_value[3:6]
            if self.joy_value[6] <= 3:
                if default_root_state[2] + self.joy_value[2] < self.gait_stop_height:
                    self.gait_type = 0
                    self.root_lin_vel_target[:] = 0
                    self.root_ang_vel_target[:] = 0
                else:
                    self.gait_type = self.joy_value[6]
            else:
                vel_total = np.linalg.norm(self.root_lin_vel_target) + np.linalg.norm(self.root_ang_vel_target)
                if vel_total < 1e-6:
                    self.gait_type = 0
                elif vel_total < 0.3:
                    self.gait_type = 1
                else:
                    self.gait_type = 2
            #quick stop
            if self.joy_value[7] == 1:
                self.quick_stop_duration+=sim_params.dt 
                self.root_lin_vel_target[0]=0.0
                self.stop_distance= self.root_pos-self.stop_distance0
                if(np.all(self.contact_target==True)):
                    self.quick_stop_=True
                    a=1
                if self.gait_type_last_stop ==0:
                    self.gait_type=0
                    self.joy_value[0:2] = 0
                # used in no lip
                # if self.quick_stop_==False:
                #     if np.linalg.norm(self.root_lin_vel[0:2])<0.02:
                #         self.gait_type=0
                #         self.quick_stop_duration-=sim_params.dt
            else:
                self.quick_stop_=False
                self.gait_changed_num=0
                self.gait_type_last_stop=self.gait_type
                self.quick_stop_first_run=True
                self.time=0
                self.stop_distance[0:3]=0
                self.stop_distance0=self.root_pos.copy()
                self.quick_stop_duration=0
            #quick stop end 
                
        else:
            self.root_lin_vel_target[:] = [0.0, 0.0, 0.0]
            self.root_ang_vel_target[:] = [0.0, 0.0, 0.0]
            self.gait_type = 0
        print("gait_type:",self.gait_type)
        self.root_pos_target = default_root_state[0:3] + com_offset + np.array([0.0, 0.0, self.joy_value[2]])
        self.root_euler_target[:] = default_root_state[3:6]

        # o = np.average(self.contact_pos_world, axis=0) + \
        #     (self.kd_root_pos * self.root_lin_vel_target) @ self.rot_mat_z.T
        o = np.average(self.contact_pos_world, axis=0)
        self.contact_pos_avg = (1 - terrain_filter_rate) * self.contact_pos_avg + terrain_filter_rate * o

        self.root_pos_target[2] += self.terrain_height
        self.root_euler_target[1] += self.terrain_pitch
        self.root_pos_target[0:2] += self.contact_pos_avg[0:2]

    def update_plan(self):
        # contact plan
        if self.counter == 0 or \
                (self.gait_type != self.gait_type_last and np.all(self.foot_state != FOOT_SW)):
            self.foot_counter_st = np.zeros(4, dtype=float)
            self.foot_counter_sw = np.zeros(4, dtype=float)
            # self.foot_counter_st_speed[:] = 1
            # self.foot_counter_sw_speed[:] = 1
            # TODO: add walking gait
            if self.gait_type == 0:
                self.foot_to_sw = []
            elif self.gait_type == 1:
                for i in range(4):
                    if self.foot_state[i] == FOOT_STD:
                        self.foot_to_sw = [i]
                        break
            elif self.gait_type == 2:
                self.foot_to_sw = [0, 3] if np.all(self.foot_state[[0, 3]] == FOOT_STD) else [1, 2]
            elif self.gait_type == 3:
                self.foot_to_sw = [0, 2] if np.all(self.foot_state[[1, 3]] == FOOT_STD) else [0, 2]
            else:
                print("Wrong gait type, set it to 0")
                self.gait_type = 0

            self.gait_type_last = self.gait_type

        # foot state machine
        foot_state_next = self.foot_state.copy()
        for i in range(4):
            if self.foot_state[i] == FOOT_ST:
                if self.foot_counter_st[i] >= count_per_phase:
                    foot_state_next[i] = FOOT_STD
            elif self.foot_state[i] == FOOT_STD:
                if np.all(self.foot_state[self.foot_to_sw] == FOOT_STD) and \
                        np.all(self.foot_state != FOOT_SW) and i in self.foot_to_sw:
                    foot_state_next[i] = FOOT_SW
            elif self.foot_state[i] == FOOT_SW:
                if (self.foot_counter_sw[i] >= count_per_phase * 0.5 and
                    np.linalg.norm(self.foot_contact_force[i]) >= contact_force_low) or \
                        (self.foot_counter_sw[i] >= count_per_phase and
                         np.linalg.norm(self.calf_contact_force[i]) >= contact_force_low) or \
                        self.foot_counter_sw[i] >= count_per_phase * 3.0:
                    foot_state_next[i] = FOOT_ST
                if self.foot_counter_sw[i] >= count_per_phase * 3.0:
                    print("Dangerous switch at {}".format(i))
            else:
                print("Wrong foot state at {}".format(i))

        self.foot_state = foot_state_next.copy()
        self.contact_target = np.isin(self.foot_state, [FOOT_ST, FOOT_STD])
        print("contact_target",self.contact_target)
        for i in range(4):
            if self.foot_state[i] == FOOT_ST:
                self.foot_counter_sw[i] = 0
            elif self.foot_state[i] == FOOT_STD:
                pass
            elif self.foot_state[i] == FOOT_SW:
                self.foot_counter_st[i] = 0

        # update contact sequence
        if self.gait_type == 0 and len(self.foot_to_sw) == 0:
            self.foot_to_sw = []
        elif self.gait_type == 1 and len(self.foot_to_sw) == 1:
            direction = np.arctan2(self.root_lin_vel_target[1], self.root_lin_vel_target[0])
            if -np.pi / 4 <= direction < np.pi / 4:
                sequence = walk_sequence_forward
            elif np.pi / 4 <= direction < np.pi * 3 / 4:
                sequence = walk_sequence_left
            elif np.pi * 3 / 4 <= direction <= np.pi or \
                    -np.pi < direction < -np.pi * 3 / 4:
                sequence = walk_sequence_backward
            elif -np.pi * 3 / 4 <= direction < -np.pi / 4:
                sequence = walk_sequence_right

            if np.all(self.foot_state[self.foot_to_sw] == FOOT_SW):
                next_index = (sequence.index(self.foot_to_sw) + 1) % 4
                self.foot_to_sw = sequence[next_index]
        elif self.gait_type == 2 and len(self.foot_to_sw) == 2:
            if np.all(self.foot_state[self.foot_to_sw] == FOOT_SW):
                sequence = [[0, 3], [1, 2]]
                next_index = (sequence.index(self.foot_to_sw) + 1) % 2
                self.foot_to_sw = sequence[next_index]
        elif self.gait_type == 3 and len(self.foot_to_sw) == 2:
            if np.all(self.foot_state[self.foot_to_sw] == FOOT_SW):
                sequence = [[0, 2], [1, 3]]
                next_index = (sequence.index(self.foot_to_sw) + 1) % 2
                self.foot_to_sw = sequence[next_index]

        # foot plan        
        self.kp_foot_x = 0.1
        self.kd_foot_x = 0.1
        self.kf_foot_x = 0.3 if self.gait_type in [0, 1] else 0.05
        self.kp_foot_y = 0.1
        self.kd_foot_y = 0.1
        self.kf_foot_y = 0.3 if self.gait_type in [0, 1] else 0.1
        self.kp_pitch_z = 0.15
        delta_foot_x = self.kp_foot_x * (self.root_pos[0] - self.root_pos_target[0]) + \
                       self.kd_foot_x * (self.root_lin_vel_rel[0] - self.root_lin_vel_target[0]) + \
                       self.kf_foot_x * self.root_lin_vel_rel[0]
        delta_foot_y = self.kp_foot_y * (self.root_pos[1] - self.root_pos_target[1]) + \
                       self.kd_foot_y * (self.root_lin_vel_rel[1] - self.root_lin_vel_target[1]) + \
                       self.kf_foot_y * self.root_lin_vel_rel[1]
        if self.use_lip:
            self.Tc=np.sqrt(self.body_height/9.80)
            T=0.3
            delta_foot_x=-(self.root_lin_vel_target[0]-self.root_lin_vel_rel[0]*np.cosh(T/self.Tc))/(np.sinh(T/self.Tc))*self.Tc
            delta_foot_y=-(self.root_lin_vel_target[1]-self.root_lin_vel_rel[1]*np.cosh(T/self.Tc))/(np.sinh(T/self.Tc))*self.Tc
            # x_max=0.25
            # if delta_foot_x>=0:
            #     delta_foot_x=np.minimum(x_max,delta_foot_x)
            # else:
            #     delta_foot_x=np.maximum(-x_max,delta_foot_x)
            # if delta_foot_y>=0:
            #     delta_foot_y=np.minimum(x_max,delta_foot_y)
            # else:
            #     delta_foot_y=np.maximum(-x_max,delta_foot_y)
        
        if (self.quick_stop_):
            delta_foot_x = self.root_lin_vel_rel[0] * np.sqrt(self.body_height / 9.80)
            delta_foot_y = self.root_lin_vel_rel[1] * np.sqrt(self.body_height / 9.80)
            x_max=0.31  
            num=np.ceil(np.maximum(np.abs(delta_foot_x/x_max),np.abs(delta_foot_y/x_max)))  
            if delta_foot_x>=x_max:
                delta_foot_x=x_max
            if delta_foot_x<=-x_max:
                delta_foot_x=-x_max
            if delta_foot_y>=x_max:
                delta_foot_y=x_max
            if delta_foot_y<=-x_max:
                delta_foot_y=-x_max    
            if self.quick_stop_first_run:
                self.lip_stop_duration=(num+1)*count_per_cycle*sim_params.dt/2-sim_params.dt*2
        
        # TODO: add height change based on terrain roll
        delta_foot_z = self.kp_pitch_z * np.sin(self.terrain_pitch) * np.array([-1, -1, 1, 1])
        delta_foot_z += -delta_foot_x * np.tan(self.terrain_pitch)
        self.foot_pos_rel_target = default_foot_pos.copy()
        self.foot_pos_rel_target[:, 0] += delta_foot_x
        self.foot_pos_rel_target[:, 1] += delta_foot_y
        self.foot_pos_rel_target[:, 2] += delta_foot_z
        self.foot_pos_rel_target[:, 2] += -self.joy_value[2]
        self.foot_pos_abs_target = self.foot_pos_rel_target @ self.rot_mat_z.T
        self.foot_pos_world_target =self.foot_pos_abs_target+ self.root_pos

    def update_command(self):
        # foot control
        foot_pos_final = self.foot_pos_abs_target @ self.rot_mat_z
        foot_pos_cur = self.foot_pos_abs @ self.rot_mat_z
        # foot_pos_cur = self.foot_pos_rel

        # TODO: clean up the following two if statements
        if self.counter == 0:
            self.foot_pos_start = foot_pos_cur

        bezier_time = np.zeros(4)
        for i in range(4):
            if self.contact_target[i]:
                self.foot_pos_start[i] = foot_pos_cur[i]
            else:
                bezier_time[i] = self.foot_counter_sw[i] / count_per_phase
            if self.foot_counter_sw[i] < count_per_phase:
                self.foot_pos_end[i] = foot_pos_final[i]
        foot_pos_target = self._get_bezier_curve(self.foot_pos_start, foot_pos_final, bezier_time)
        # foot_pos_target = foot_pos_final

        foot_pos_error = foot_pos_target - foot_pos_cur
        foot_force_kin = self.kp_kin * foot_pos_error
        foot_force_kin_flat = foot_force_kin.flatten()

        # avoid straight knee singularity
        A_aug = np.vstack((self.foot_jaco, 0.1 * np.eye(12)))
        b_aug = np.hstack((self.km_kin * foot_force_kin_flat, np.zeros(12)))
        torque_kin = np.linalg.lstsq(A_aug, b_aug, rcond=-1)[0]
        # torque_kin = np.linalg.solve(self.foot_jaco, self.km_kin * foot_force_kin_flat)
        # torque_kin = self.foot_jaco.T @ foot_force_kin_flat

        # touch down control
        foot_pos_error = self.foot_pos_end - foot_pos_cur
        foot_force_down = self.kp_kin * foot_pos_error
        foot_force_down[:, 2] = -contact_force_high
        foot_force_down_flat = foot_force_down.flatten()
        torque_down = np.linalg.solve(self.foot_jaco, self.km_kin * foot_force_down_flat)

        # swing foot reaches limit
        if np.any(self.foot_counter_sw >= count_per_phase):
            self.root_pos_delta_z += self.root_vel_delta_z * sim_dt
        else:
            self.root_pos_delta_z *= (1 - root_height_filter_rate)
        self.root_pos_target[2] += self.root_pos_delta_z

        # grf control
        foot_force_grf = -self._root_control()
        foot_force_grf_rel = foot_force_grf @ self.rot_mat
        foot_force_grf_rel_flat = foot_force_grf_rel.flatten()
        torque_grf = self.foot_jaco.T @ foot_force_grf_rel_flat

        # merge torque
        for i in range(4):
            if self.contact_target[i]:
                self.torque[3 * i:3 * i + 3] = torque_grf[3 * i:3 * i + 3]
            else:
                if self.foot_counter_sw[i] < count_per_phase:
                    self.torque[3 * i:3 * i + 3] = torque_kin[3 * i:3 * i + 3]
                else:
                    self.torque[3 * i:3 * i + 3] = torque_down[3 * i:3 * i + 3]

        # self.torque = torque_kin
        # self.torque = torque_down
        # self.torque = torque_grf

        # hip gravity compensation
        self.torque += np.array([+1.1, 0, 0, -1.1, 0, 0, +1.1, 0, 0, -1.1, 0, 0])

        # DEBUG joint PD
        # kp = np.array([100.0, 100.0, 100.0,
        #                100.0, 100.0, 100.0,
        #                100.0, 100.0, 100.0,
        #                100.0, 100.0, 100.0, ])
        # kp[0+6] = visual_ros.user_input.data[0]
        # kp[1+6] = visual_ros.user_input.data[1]
        # kp[2+6] = visual_ros.user_input.data[2]

        if state_zero_torque:
            self.torque[:] = 0

        self.torque = np.clip(self.torque, -dof_props["effort"] + self.joint_vel * dof_props["damping"],
                              dof_props["effort"] + self.joint_vel * dof_props["damping"])
    
        torch_torque = torch.from_numpy(self.torque).to(torch.float)
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torch_torque))

    def update_counter(self):
        self.counter += 1
        for i in range(4):
            if self.foot_state[i] == FOOT_ST:
                self.foot_counter_st[i] += 1#self.foot_counter_st_speed[i]
            elif self.foot_state[i] == FOOT_SW:
                self.foot_counter_sw[i] += 1#self.foot_counter_sw_speed[i]
        # global count_per_cycle, count_per_phase
        # count_per_cycle = np.clip(visual_ros.user_input.data[7], 60, 150)
        # count_per_phase = count_per_cycle / 2

    def check_termination(self):
        max_distance = np.inf
        max_height = 10
        max_pitch_roll = np.deg2rad(60)
        max_joint_vel = 30

        xy_distance_check = np.linalg.norm(self.root_pos[0:2]) > max_distance
        z_height_check = np.abs(self.root_pos[2]) > max_height
        orientation_check = np.any(np.abs(self.root_euler[0:2]) > max_pitch_roll)
        joint_vel_check = np.any(np.abs(self.joint_vel) > max_joint_vel)

        if np.any([xy_distance_check, z_height_check, orientation_check, joint_vel_check]):
            self.reset()

    def draw_lines(self):
        # draw foot connection
        p1_index = [foot_index[0], foot_index[0], foot_index[0], foot_index[1], foot_index[1], foot_index[2]]
        p2_index = [foot_index[1], foot_index[2], foot_index[3], foot_index[2], foot_index[3], foot_index[3]]

        for i in range(6):
            p1 = gym.get_rigid_transform(env, p1_index[i])
            p2 = gym.get_rigid_transform(env, p2_index[i])
            color = gymapi.Vec3(*color_green)
            gymutil.draw_line(p1.p, p2.p, color, gym, viewer, env)

        # draw root that projects on the ground
        root_project_pose = gym.get_rigid_transform(env, 0) #0:base
        root_project_pose.p.z = foot_radius + self.terrain_height
        gymutil.draw_lines(sphere_red, gym, viewer, env, root_project_pose)

        # draw estimate root that projects on the ground  self.root_pos_des_abs
        root_est_project_pose = gymapi.Transform()
        # root_est_project_pose.p = gymapi.Vec3(*self.contact_pos_avg)

        # Draw GRF
        for i in range(4):
            pi = gym.get_rigid_transform(env, foot_index[i])
            color = gymapi.Vec3(*color_red)
            gymutil.draw_line(pi.p, pi.p + gymapi.Vec3(*(0.002 * self.foot_contact_force[i])), color, gym, viewer, env)

            # If in contact, draw red if not in contact draw blue
            if (np.linalg.norm(self.foot_contact_force[i]) > 0):
                gymutil.draw_lines(sphere_red, gym, viewer, env, pi)
            else:
                gymutil.draw_lines(sphere_blue, gym, viewer, env, pi)

    def _root_control(self):
        # TODO: fix the 180 degree bug
        euler_error = self.root_euler_target - self.root_euler
        root_acc_target = np.zeros(6)
        #test
        if self.use_lip:
            # if (self.contact_target_last[0] !=self.contact_target[0] and self.contact_target_last[3] !=self.contact_target[3]) or (self.contact_target_last[1] !=self.contact_target[1] and self.contact_target_last[2] !=self.contact_target[2]):
            # if (not np.array_equal(self.contact_target_last,self.contact_target)) and not np.all(self.contact_target==True):
            if np.all(self.contact_target==True):
                self.Tc=np.sqrt(self.body_height/9.80)
                T=count_per_cycle*sim_params.dt/2
                root_vel_des=self.rot_mat_z@self.root_lin_vel_target
                self.root_p0=(root_vel_des-self.root_lin_vel*np.cosh(T/self.Tc))/(np.sinh(T/self.Tc))*self.Tc
                self.root_p0[2]=0
                self.root_des=-self.root_p0+self.root_pos
                self.root_des[2]=self.body_height 
                self.root_v0=self.root_lin_vel.copy()
                self.root_v0[2]=0
                print("self.root_time",self.root_time)
                self.root_time=0          
            self.root_time+=sim_params.dt
            root_pos_abs=np.zeros(3)
            root_pos_abs=self.root_p0*np.cosh(self.root_time/self.Tc)+self.Tc*self.root_v0*np.sinh(self.root_time/self.Tc)
            # print("root_pos_abs",root_pos_abs)
            self.root_pos_target[0:2] =root_pos_abs[0:2]+self.root_des[0:2]   
            self.root_pos_target[2]=self.body_height       
            root_vel_abs=np.zeros(3)    
            root_vel_abs= self.root_p0/self.Tc*np.sinh(self.root_time/self.Tc)+self.root_v0*np.cosh(self.root_time/self.Tc) #world frame
            # print("root_vel_abs",root_vel_abs)
            self.root_lin_vel_target=self.rot_mat_z.T@root_vel_abs
            root_acc_lip=np.zeros(3)
            root_acc_lip =root_pos_abs/(self.Tc**2) #+1000*(self.root_pos_des_abs-self.root_pos)
            root_acc_lip[2]=0
            root_acc_target[0:3] += root_acc_lip
            # root_acc_target[0:3] +=5000*(self.root_pos_target - self.root_pos)
            self.contact_target_last=self.contact_target
        #test end

        if(self.quick_stop_):
            #p0 v0 
            if self.quick_stop_first_run: 
                self.vel0=self.root_lin_vel[0:2].copy() #in world frame
                self.Tc=np.sqrt(self.body_height/9.80)
                self.root_pos_des_abs[0:2] = self.root_lin_vel[0:2] * self.Tc+self.root_pos[0:2]
                self.pos0=-self.vel0 * self.Tc #in world frame
                self.time=0
                self.quick_stop_first_run=False 
            self.time+=sim_params.dt
            if self.time<=self.lip_stop_duration:
                print("self.time",self.time)
                root_pos_abs=np.zeros(3)
                root_pos_abs[0:2]=self.pos0*np.cosh(self.time/self.Tc)+self.Tc*self.vel0*np.sinh(self.time/self.Tc)
                self.root_pos_target[0:2] =root_pos_abs[0:2]+self.root_pos_des_abs[0:2]   
                self.root_pos_target[2]=self.body_height  
                root_vel_abs=np.zeros(3)         
                root_vel_abs[0:2]= self.pos0/self.Tc*np.sinh(self.time/self.Tc)+self.vel0*np.cosh(self.time/self.Tc) 
                self.root_lin_vel_target=self.rot_mat_z.T@root_vel_abs
                self.root_acc_quick_stop =root_pos_abs/(self.Tc**2) 
                self.root_acc_quick_stop[2]=0
                root_acc_target[0:3] += self.root_acc_quick_stop
            else:
                # print("self.time",self.time)
                self.time-=sim_params.dt
                self.gait_type=0
                self.root_pos_target[0:2] =self.root_pos_des_abs[0:2]
                self.root_pos_target[2]=self.body_height
                self.root_lin_vel_target[0:3]=0
                root_acc_target[0:3]=0
                self.root_acc_quick_stop[0:3]=0
                self.quick_stop_duration-=sim_params.dt
            self.gait_type_last_stop=self.gait_type
            self.contact_target_last =self.contact_target.copy()
        #quick stop end
        root_acc_target[0:3] += self.kp_root_lin * (self.root_pos_target - self.root_pos)
        root_acc_target[0:3] += (self.kd_root_lin * (self.root_lin_vel_target - self.root_lin_vel_rel)) @ \
                                self.rot_mat_z.T
        root_acc_target[3:6] += self.kp_root_ang * euler_error
        root_acc_target[3:6] += self.kd_root_ang * (self.root_ang_vel_target - self.root_ang_vel_rel)

        gravity = 9.8 * total_mass
        root_acc_target[2] += gravity

        A = visual_ros.user_input.data[4]
        omega = visual_ros.user_input.data[5]
        theta = visual_ros.user_input.data[6]
        # omega = 1.5 / (count_per_cycle * sim_dt)
        root_acc_x_regulation = 0
        for i in range(4):
            if self.contact_target[i]:
                root_acc_x_regulation -= A * np.sin(2 * np.pi * (self.foot_counter_st[i] * sim_dt * omega + theta))
        root_acc_x_regulation /= np.sum(self.contact_target)
        visual_ros.append_data(root_acc_x_regulation)  # 51

        root_acc_target[0] += root_acc_x_regulation

        inv_inertia_mat = np.zeros([6, 12])
        inv_inertia_mat[0:3, :] = np.tile(np.eye(3), 4)
        for i in range(4):
            inv_inertia_mat[3:6, i * 3: i * 3 + 3] = self.rot_mat_z.T @ skew(self.foot_pos_abs[i, :] - com_offset)

        acc_weight = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        grf_weight = 1e-3
        grf_diff_weight = 1e-2
        grf = self.qp.casadi_solve(inv_inertia_mat,
                                   root_acc_target,
                                   acc_weight,
                                   grf_weight,
                                   grf_diff_weight,
                                   self.grf_last,
                                   self.contact_target)
        self.grf_last = grf.flatten()  # Store this solve to use in the next solve
        return grf

    def _get_bezier_curve(self, foot_pos_start, foot_pos_final, bezier_time):
        foot_pos_target = np.zeros([4, 3])

        for i in range(4):
            bezier_x = np.array([foot_pos_start[i, 0],
                                 foot_pos_start[i, 0],
                                 foot_pos_final[i, 0],
                                 foot_pos_final[i, 0],
                                 foot_pos_final[i, 0]])
            foot_pos_target[i, 0] = bezier_curve(bezier_time[i], bezier_x)

        for i in range(4):
            bezier_y = np.array([foot_pos_start[i, 1],
                                 foot_pos_start[i, 1],
                                 foot_pos_final[i, 1],
                                 foot_pos_final[i, 1],
                                 foot_pos_final[i, 1]])
            foot_pos_target[i, 1] = bezier_curve(bezier_time[i], bezier_y)

        for i in range(4):
            bezier_z = np.array([foot_pos_start[i, 2],
                                 foot_pos_start[i, 2],
                                 foot_pos_final[i, 2],
                                 foot_pos_final[i, 2],
                                 foot_pos_final[i, 2]])
            bezier_z[1] += 0.0
            bezier_z[2] += np.minimum(0.4, 3 * (default_root_state[2] + self.joy_value[2] - self.gait_stop_height))
            foot_pos_target[i, 2] = bezier_curve(bezier_time[i], bezier_z)

        return foot_pos_target


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def bezier_curve(alpha, param):
    # 4th order bezier, 5 coefficient
    degree = 4
    coefficient = [1, 4, 6, 4, 1]
    y = 0
    for i in range(degree + 1):
        y += coefficient[i] * np.power(alpha, i) * np.power(1 - alpha, degree - i) * param[i]
    return y


gym = gymapi.acquire_gym()

# create sim
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 200
sim_params.substeps = 1
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.num_threads = 1
sim_params.physx.use_gpu = False
sim_params.physx.solver_type = 0  # Switched since solver type 1 has bad velocity values
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# create ground
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# create viewer
camera_properties = gymapi.CameraProperties()
camera_properties.width = 1600
camera_properties.height = 900
viewer = gym.create_viewer(sim, camera_properties)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "RESET")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "ESCAPE")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_H, "HOLD")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "PERTURB")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "STOP")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_O, "ZERO_TORQUE")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "TURN_RIGHT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "UP")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "DOWN")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "TURN_LEFT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "LEFT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "RIGHT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_U, "STAND")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "SIT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "0")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "1")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "2")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "3")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "SMART_GAIT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "QUICK_STOP")

# load asset
asset_root = "/home/yu/workspace/python_work/model/laikago/"
asset_name = "laikago_obj"
asset_file = "urdf/" + asset_name + ".urdf"
terrain_file = "slope.urdf"
ros_topic = "laikago"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.use_mesh_materials = True
asset_options.angular_damping = 0.0

terrain_options = gymapi.AssetOptions()
terrain_options.fix_base_link = True
terrain_options.use_mesh_materials = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
terrain = gym.load_asset(sim, asset_root, terrain_file, terrain_options)

dof_dict = gym.get_asset_dof_dict(asset)
rigid_body_dict = gym.get_asset_rigid_body_dict(asset)
joint_dict = gym.get_asset_joint_dict(asset)

# position the camera
cam_pos = gymapi.Vec3(2.0, 1.0, 0.4)
cam_target = gymapi.Vec3(1.0, 0.0, 0.4)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# create env
env_lower = gymapi.Vec3()
env_upper = gymapi.Vec3()
num_per_row = 0
env = gym.create_env(sim, env_lower, env_upper, num_per_row)

pose = gymapi.Transform()
actor_handle = gym.create_actor(env, asset, pose, asset_name, 0, 1, 0)
# set joint
dof_props = gym.get_actor_dof_properties(env, actor_handle)
dof_props["stiffness"].fill(0)
dof_props["damping"].fill(1.0)
dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
gym.set_actor_dof_properties(env, actor_handle, dof_props)

terrain_handle = gym.create_actor(env, terrain, pose, "terrain", 0, 0, 0)
for i in range(gym.get_actor_rigid_body_count(env, terrain_handle)):
    gym.set_rigid_body_color(env, terrain_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.4, 0.2, 0.1))

rigid_body_props = gym.get_actor_rigid_body_properties(env, actor_handle)

gym.prepare_sim(sim)

# default state
sim_dt = sim_params.dt
count_per_cycle = 100
count_per_phase = count_per_cycle / 2

init_root_state = np.zeros(13)
init_root_state[0:3] = [0.0, 0.0, 0.6]
init_root_state[3:7] = [0.0, 0.0, 0.0, 1.0]
init_root_state[7:10] = [0.0, 0.0, 0.0]

com_offset = np.array([0.00, 0, 0.0])

default_root_state = init_root_state.copy()
default_root_state[0:3] = [0.0, 0.0, 0.45]

knee_sign = np.sign(dof_props["lower"] + dof_props["upper"])
default_dof_position = np.array([0.0, 0.6, 1.2,
                                 0.0, 0.6, 1.2,
                                 0.0, 0.6, 1.2,
                                 0.0, 0.6, 1.2]) * knee_sign

default_dof_velocity = np.zeros(12)

default_foot_pos = np.array([[+0.2375, +0.12, -default_root_state[2] - 0.01],
                             [+0.2375, -0.12, -default_root_state[2] - 0.01],
                             [-0.2375, +0.12, -default_root_state[2] - 0.01],
                             [-0.2375, -0.12, -default_root_state[2] - 0.01]])
default_foot_pos[:] += com_offset

init_terrain_state = np.zeros(13)
init_terrain_state[0:3] = [0.0, 0.0, -10.05]
init_terrain_state[3:7] = [0.0, 0.0, 0.0, 1.0]

calf_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
calf_index = [rigid_body_dict[name] for name in calf_name]

foot_name = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
foot_index = [rigid_body_dict[name] for name in foot_name]
foot_radius = 0.0265

rigid_body_names = gym.get_actor_rigid_body_names(env, actor_handle)
for i in range(len(rigid_body_props)):
    rigid_body_props[i].mass *= 1
    print(rigid_body_names[i], "\tmass = ", rigid_body_props[i].mass, "\tcom = ", rigid_body_props[i].com)

total_mass = np.sum([rigid_body_props[i].mass for i in range(len(rigid_body_props))])
print("total mass = ", total_mass)
gym.set_actor_rigid_body_properties(env, actor_handle, rigid_body_props)

contact_force_low = 20
contact_force_high = 50

FOOT_ST = 0
FOOT_STD = 1
FOOT_SW = 2

walk_sequence_forward = [[0], [3], [1], [2]]
walk_sequence_backward = [[0], [2], [1], [3]]
walk_sequence_left = [[0], [3], [2], [1]]
walk_sequence_right = [[0], [1], [2], [3]]

terrain_filter_rate = 10.0 * sim_dt
root_height_filter_rate = 1.0 * sim_dt

color_red = (1, 0, 0)
color_green = (0, 1, 0)
color_blue = (0, 0, 1)
color_grey = (0.2, 0.2, 0.2)
sphere_red = gymutil.WireframeSphereGeometry(foot_radius, 12, 12, gymapi.Transform(), color=color_red)
sphere_green = gymutil.WireframeSphereGeometry(foot_radius, 12, 12, gymapi.Transform(), color=color_green)
sphere_blue = gymutil.WireframeSphereGeometry(foot_radius, 12, 12, gymapi.Transform(), color=color_blue)
sphere_grey = gymutil.WireframeSphereGeometry(foot_radius, 12, 12, gymapi.Transform(), color=color_grey)

# init robot state
robot_state = RobotState()
visual_ros = VisualRos(ros_topic)
quick_stop_ros = VisualRos("quick_stop")

# ring buffer
joint_pos_buffer = deque(maxlen=480)
joint_vel_buffer = deque(maxlen=480)
joint_torque_buffer = deque(maxlen=480)

# init tensor
root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
jacobian_tensor = gym.acquire_jacobian_tensor(sim, asset_name)
net_contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)

root_state = gymtorch.wrap_tensor(root_state_tensor)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)
jacobian = gymtorch.wrap_tensor(jacobian_tensor)
net_contact_force = gymtorch.wrap_tensor(net_contact_force_tensor)

dof_position = dof_state[:, 0]
dof_velocity = dof_state[:, 1]

# init attractor
attractor_active = False
attractor_properties = gymapi.AttractorProperties()
attractor_properties.rigid_handle = gym.get_actor_root_rigid_body_handle(env, actor_handle)
attractor_properties.target = gymapi.Transform(gymapi.Vec3(*init_root_state[0:3]),
                                               gymapi.Quat(*init_root_state[3:7]))
attractor_properties.stiffness = 1e6 if attractor_active else 0
attractor_properties.damping = 1e4 if attractor_active else 0
attractor_properties.axes = gymapi.AXIS_ALL
attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)

# reset
state_zero_torque = False
robot_state.reset()
robot_state.body_height=-np.mean(default_foot_pos[:,2])
# move viewer window
# window_jumper.main()
counter_stop=10
quick_key_sign=False
# simulate
while not gym.query_viewer_has_closed(viewer):
    begin_time=time.time()
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    cam_pos.x = 1.0 + robot_state.root_pos[0]
    cam_pos.y = 1.0 + robot_state.root_pos[1]
    cam_pos.z= 0.5
    cam_target.x = 0.0 + robot_state.root_pos[0]
    cam_target.y = 0.0 + robot_state.root_pos[1]
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # refresh tensor
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim)

    # feedback control
    robot_state.update_state()
    robot_state.update_output()
    robot_state.update_user()
    robot_state.update_plan()
    robot_state.update_command()
    robot_state.update_counter()
    robot_state.check_termination()
    visual_ros.publish_data()
    quick_stop_ros.publish_data()

    # check for keyboard events
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "RESET" and evt.value > 0:
            robot_state.reset()
        elif evt.action == "ESCAPE" and evt.value > 0:
            sys.exit()
        elif evt.action == "HOLD" and evt.value > 0:
            attractor_active ^= True
            attractor_properties = gym.get_attractor_properties(env, attractor_handle)
            attractor_properties.stiffness = 1e6 if attractor_active else 0
            attractor_properties.damping = 1e4 if attractor_active else 0
            attractor_properties.axes = gymapi.AXIS_ALL
            gym.set_attractor_properties(env, attractor_handle, attractor_properties)
        elif evt.action == "PERTURB" and evt.value > 0:
            gym.apply_body_force_at_pos(env, rigid_body_dict["base"], gymapi.Vec3(0, 1000, 0))
            print("apply side push")
        elif evt.action == "STOP" and evt.value > 0:
            robot_state.joy_value[0:6] = 0
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "ZERO_TORQUE" and evt.value > 0:
            state_zero_torque ^= True
        elif evt.action == "TURN_LEFT" and evt.value > 0:
            robot_state.joy_value[5] = np.minimum(2.0, robot_state.joy_value[5] + 0.1)
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "UP" and evt.value > 0:
            robot_state.joy_value[0] = np.minimum(2.0, robot_state.joy_value[0] + 0.1)
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "DOWN" and evt.value > 0:
            robot_state.joy_value[0] = np.maximum(-1.0, robot_state.joy_value[0] - 0.1)
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "TURN_RIGHT" and evt.value > 0:
            robot_state.joy_value[5] = np.maximum(-2.0, robot_state.joy_value[5] - 0.1)
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "LEFT" and evt.value > 0:
            robot_state.joy_value[1] = np.minimum(0.5, robot_state.joy_value[1] + 0.1)
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "RIGHT" and evt.value > 0:
            robot_state.joy_value[1] = np.maximum(-0.5, robot_state.joy_value[1] - 0.1)
            print("joy_value = {}".format(robot_state.joy_value))
        elif evt.action == "STAND" and evt.value > 0:
            robot_state.joy_value[2] = np.minimum(0.1, robot_state.joy_value[2] + 0.02)
        elif evt.action == "SIT" and evt.value > 0:
            robot_state.joy_value[2] = np.maximum(-0.4, robot_state.joy_value[2] - 0.02)
        elif evt.action == "0" and evt.value > 0:
            robot_state.joy_value[6] = 0
        elif evt.action == "1" and evt.value > 0:
            robot_state.joy_value[6] = 1
        elif evt.action == "2" and evt.value > 0:
            robot_state.joy_value[6] = 2
        elif evt.action == "3" and evt.value > 0:
            robot_state.joy_value[6] = 3
        elif evt.action == "SMART_GAIT" and evt.value > 0:
            robot_state.joy_value[6] = 4
        elif evt.action == "QUICK_STOP" and evt.value > 0:
            if quick_key_sign==True:
                quick_key_sign=False
            else:
                quick_key_sign=True
    #quick stop
    if np.all(robot_state.contact_target==True):
        counter_stop=0
    if quick_key_sign==True and counter_stop==1:
        robot_state.joy_value[7] = 1
        print("enter quick stop")
    if quick_key_sign==False:
        robot_state.joy_value[7] = 0
    counter_stop+=1
    # update the viewer
    gym.clear_lines(viewer)
    robot_state.draw_lines()

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    end_time=time.time()
    print("time",end_time-begin_time)
    while end_time<begin_time+sim_dt:
        end_time=time.time()

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
