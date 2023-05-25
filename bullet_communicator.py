#!/usr/bin/env python3

import os
import sys
import numpy as np
import math
import time
import copy
import ctypes
from sim_msg import communication_combine, locomotion
import lcm
import json
from bullet_interface import BulletInterface

np.set_printoptions(linewidth=1000, precision=3)

curPath = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(curPath)[0]

is_shutdown = False

class BulletCommunicator:
    def __init__(self, settings: dict):
        self.settings = settings
        self.interface = BulletInterface(settings)

    def run(self):
        counter = 0
        effort = [0.0]*12

        while not is_shutdown:
            begin_time = time.time()
            self.interface.handle_keyboard()

            # update robot states
            imu_data, leg_data, foot_data, ground_truth = self.interface.get_states()
            # acceleration = imu_data[0:3]
            # # convert JPL convention to Hamilton convention
            # quaternion = imu_data[3:7]
            # gyroscope = imu_data[7:10]
            # joint_position = leg_data[0:12]
            # joint_velocity = leg_data[12:24]
            # torque_reading = [0.0]*12
            # foot_contact_force =  foot_data

            lcm_client.publish_communication_combine(imu_data, leg_data, foot_data, ground_truth)
            lcm_client.update_sub_state()
            effort = lcm_client.subscribe_locomotion(leg_data)

            # set locomotion commands
            self.interface.set_forces(effort)

            # apply attractor
            self.interface.apply_attractor_force()

            # set camera view
            if counter % 10 == 0:
                self.interface.set_visualizer_camera()

            # view contact force
            # if counter % 100 == 0:
            #     self.interface.view_foot_contacts()

            # run simulation
            self.interface.run()

            counter += 1

            current_time = time.time()
            end_time  = begin_time + self.settings['cycle_time']*self.interface.read_sim_speed()
            while current_time < end_time:
                current_time = time.time()

class LCMClient:
    def __init__(self):
        self.lc = lcm.LCM()
        self.input_data = {}

    def subscribe(self):
        self.subscription = self.lc.subscribe("/locomotion", self.on_receive_message)
        self.subscription.set_queue_capacity(1)

    def on_receive_message(self, channel, message):
        self.input_data = locomotion.decode(message)

    def subscribe_locomotion(self, leg_data):
        if self.input_data == {}:
            return [0.0]*12
        torque = np.array(self.input_data.joint_torque)
        kp = np.array(self.input_data.joint_stiffness)
        kd = np.array(self.input_data.joint_damping)
        position_command = np.array(self.input_data.joint_position)
        velocity_command = np.array(self.input_data.joint_velocity)
        current_position = np.array(leg_data[0:12])
        current_velocity = np.array(leg_data[12:24])
        effort = torque + kp * (position_command - current_position) + kd * (velocity_command - current_velocity)
        return effort

    # def publish_bullet_states(self, imu_data, leg_data, foot_data, ground_truth):
    #     output_data = {}
    #     output_data["acceleration"] = imu_data[0:3]
    #     # convert JPL convention to Hamilton convention
    #     output_data["quaternion"] = [imu_data[6]] + imu_data[3:6]
    #     output_data["gyroscope"] = imu_data[7:10]
    #     output_data["joint_position"] = leg_data[0:12]
    #     output_data["joint_velocity"] = leg_data[12:24]
    #     output_data["torque_reading"] = [0.0]*12
    #     output_data["foot_contact_force"] =  foot_data
    #     output_data["body_acceleration"] = [ground_truth['body_orientation'][3]] + list(ground_truth['body_orientation'][:3])
    #     output_data["body_omega"] = ground_truth['body_angular_velocity']
    #     # convert JPL convention to Hamilton convention
    #     output_data["orientation"] = [ground_truth['body_orientation'][3]] + list(ground_truth['body_orientation'][:3])
    #     output_data["position"] = ground_truth['body_position']
    #     output_data["body_velocity"] = ground_truth['body_linear_velocity']
    #     message = msgpack.packb(output_data, use_bin_type=True)
    #     self.lc.publish("/bullet/states", message)

    def publish_communication_combine(self, imu_data, leg_data, foot_data, ground_truth):
        message = communication_combine()
        message.tick = time.time()
        message.root_acceleration_in_body = imu_data[0:3]
        message.root_euler = [0.0]*3
        message.root_quaternion = imu_data[3:7]
        message.root_angular_velocity_in_body = imu_data[7:10]
        message.joint_position = leg_data[0:12]
        message.joint_velocity = leg_data[12:24]
        message.joint_acceleration = [0.0]*12
        message.torque_reading = leg_data[24:36]
        message.foot_contact_force = foot_data
        self.lc.publish("/communication/combine", message.encode())

    def update_sub_state(self):
        # self.lc.handle_timeout(0)  # ms
        self.lc.handle_timeout(1000)  # ms
        # self.lc.handle()

if __name__ == '__main__':
    settings = dict(
        # enable_robot='et1_beta',
        # enable_robot='px2_et1',
        enable_robot='px2_ot',
        # enable_robot='et2',
        cycle_time=0.002,
        is_display=True,
        path=path
    )
    # json_file_name = sys.argv[2]
    # with open(json_file_name, 'r') as file:
    #     config = json.load(file)

    # if 'enable_robot' in config:
    #     settings['enable_robot'] = config['enable_robot']
    # if 'default_cycle_time' in config:
    #     settings['cycle_time'] = config['default_cycle_time']

    bullet_communicator = BulletCommunicator(settings)

    lcm_client = LCMClient()
    lcm_client.subscribe()

    bullet_communicator.run()
