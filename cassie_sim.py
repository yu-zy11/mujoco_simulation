import numpy as np
import yaml
import os
import pybullet as p
import pybullet_data
import random
import time
import sys
from visual_ros1 import  VisualRos
import rospy
import threading
# import lcm_message.lcm_msgs 
class robotSim():
    def __init__(self):
        self.run_path=os.path.dirname(__file__)
        self.yaml_path=os.path.join(self.run_path,'config','config.yaml')
        self.loadConfig()
        self.jointsID=[]
        self.initSimulation()
        self.joint_kp=[0]*self.joint_num
        self.joint_kd=[0]*self.joint_num
        self.tau_ff=[0]*self.joint_num
        self.q_des=[0]*self.joint_num
        self.dq_des=[0]*self.joint_num
        self.ddq_des=[0]*self.joint_num
        self.joint_position=[0]*self.joint_num
        self.joint_velocity=[0]*self.joint_num
        self.joint_torque=[0]*self.joint_num
        self.joint_command=[]
        self.root_pos=[0]*3
        self.root_vel=[0]*3
        self.root_omega=[0]*3
        self.root_quaternion=[0]*4 #x y z w
        self.root_rpy=[0]*4 #x y z w
        self.rotm_body2world=np.zeros((3,3))

    def loadConfig(self):
        if os.path.exists(self.yaml_path):
            with open(file=self.yaml_path, mode="rb") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.sim_mode = config.get("sim_mode")
                self.urdf_name = config.get("urdf_name")
                self.useFixedBase=config.get("useFixedBase")
                self.message_type=config.get("message_channle")
                self.sim_step = config.get("sim_step")
                self.basePosition=config.get("basePosition")
                self.baseOrientation=config.get("baseOrientation")
                self.gravity=config.get("gravity")
                print("loading Parameter from yaml in robotSim finished")
        else:
            print("wrong!,can not find yaml's path in robotSim!")
            os.abort()

    def updateRobotStates(self):
        self.root_vel,self.root_omega = p.getBaseVelocity(self.robot_id)
        self.root_pos, self.root_quaternion = p.getBasePositionAndOrientation(self.robot_id)
        self.root_rpy = p.getEulerFromQuaternion(self.root_quaternion)
        joint_states = p.getJointStates(self.robot_id, self.jointsID)
        for i in np.arange(len(self.jointsID)):
            self.joint_position[i]=joint_states[i][0]
            self.joint_velocity[i]=joint_states[i][1]
        rotation_matrix = p.getMatrixFromQuaternion(self.root_quaternion) # rotation matrix from world to body
        self.rotm_body2world = np.array (rotation_matrix).reshape(3, 3)
        # velocity=self.rotm_body2world.transpose()@np.array(self.root_velocity[0]).reshape(3,1)
        # angular_velocity=self.rotm_body2world.transpose()@np.array(self.root_velocity[1]).reshape(3,1)


    def publishStates(self):
        client_ros=VisualRos()
        # root_state
        for i in range(len(self.root_pos)):#0-2 body position
            client_ros.append_root_data(self.root_pos[i]) 
        for i in range(len(self.root_rpy)):#3-5 body euler
            client_ros.append_root_data(self.root_rpy[i])
        for i in range(len(self.root_vel)):#6-8 body velocity
            client_ros.append_root_data(self.root_vel[i])
        for i in range(len(self.root_omega)):#9-11 body angular velocity
            client_ros.append_root_data(self.root_omega[i])
        #joint states
        for i in range(len(self.joint_position)):#0-14
             client_ros.append_joint_data(self.joint_position[i])
        for i in range(len(self.joint_velocity)):#15-28
             client_ros.append_joint_data(self.joint_velocity[i])   
        client_ros.publish_data()      

    def updateCommand(self):
        # self.joint_command=self.msgs_channel.JointCommand
        print(self.joint_command)        
    
    def runRobotControll(self):
        pi=3.1415926
        joint_kp=[200,100,400,50,100,100,10, 100,100,100,100,100,100,0 ]
        joint_kd=[12,0.5,3,0.5,0.5,0.5,0.01,  0.5,0.5,0.5,0.5,0.5,0.5,0]
        tau_ff=[0]*self.joint_num
        # q_des=[0.0, 0.0, pi/3, -pi/2, -pi/8 ,pi/2 ,0, 0,0,0,0,0,0,0]
        q_des=[0.0, 0.0, pi/3, -pi/2, -pi/8 ,pi/2 ,-pi/2,   0,0,0,0,0,0,0]
        dq_des=[0]*self.joint_num
        q_data=self.joint_position
        dq_data=self.joint_velocity
        
        tau=[1]*self.joint_num
        for i in np.arange(self.joint_num):
            tau[i]=joint_kp[i]*(q_des[i]-q_data[i])+joint_kd[i]*(dq_des[i]-dq_data[i])+tau_ff[i]
        # tau[6]=0.0
        tau[13]=0.0
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,jointIndices=self.jointsID,controlMode=p.TORQUE_CONTROL,forces=tau)   
         
    def initSimulation(self): #the use of function in pybullet refer to https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit# 
        physicsClient=p.connect(p.GUI)                       #connect to GUI physics server
        if not p.isConnected:
            print("wrong!,can not connect to pybullet physics server")
            os.abort()  
        # loadURDF
        self.urdf_path=os.path.join(self.run_path,'./model/cassie_description/urdf/cassie.urdf') 
        # self.urdf_path=os.path.join(self.run_path,'./model/atlas_description/urdf/atlas_v4_with_multisense.urdf') 
        self.robot_id=p.loadURDF(self.urdf_path,basePosition=[0,0,1.2],baseOrientation=[0, 0, 0 ,1], useFixedBase=True,flags=p.URDF_MERGE_FIXED_LINKS)
        if self.robot_id<0:
                print("loading URDF failed!")
                os.abort()
        #set configuration for simulation
        p.setGravity(self.gravity[0],self.gravity[1],self.gravity[2])
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #use pybullet data package
        p.setTimeStep(self.sim_step)
        #get joint id
        j_num=p.getNumJoints(self.robot_id)
        self.jointsID=[]
        for j in range(j_num):
            joint_info = p.getJointInfo(self.robot_id, j)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.jointsID.append(j)
        self.joint_num=len(self.jointsID)
        for joint_id in self.jointsID:#disable velocity control mode ,and reset joint control mode to torque control
            p.setJointMotorControl2(self.robot_id, joint_id,controlMode=p.VELOCITY_CONTROL, force=0)      
        print("set simulation step:",self.sim_step)           
        print("init simulation finished!")  
        p.resetDebugVisualizerCamera(cameraDistance=2.0,
                             cameraYaw=45,
                             cameraPitch=-30,
                             cameraTargetPosition=[0.0, 0.0, 1.0])  
        pi=3.1415926
        targetPosition=[0.0, 0.0, pi/3, -pi/2, -pi/8 ,pi/2 ,pi, 0,0,0,0,0,0,0]
        for i in range(self.joint_num):
             p.resetJointState(self.robot_id,i,targetPosition[i])




if __name__=='__main__':
    print("start simulation ")
    Rsim=robotSim()
    # Rsim.runSimulation() 
    while p.isConnected:
        begin_time = time.time()
        Rsim.updateRobotStates()
        Rsim.publishStates()
        Rsim.runRobotControll()
        p.stepSimulation()
        current_time=time.time()
        end_time=begin_time+Rsim.sim_step
        run_time=current_time-begin_time
        if run_time>Rsim.sim_step:
            print("run_time",run_time)
            print("running time is bigger than simulation step,please change sim_step ")
        while current_time<end_time:
            current_time=time.time()