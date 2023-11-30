from simulator import MujocoSimulator
from simulator import State
import time
import math
import random

model_xml = "ls_dog105/scene.xml"
sim = MujocoSimulator(model_xml)
robot_state=State()

def initSimulation():
    sim.initSimulator()
    sim.initController()

def getRobotState():
#    robot_state= sim.getRobotState()
   return sim.getRobotState()

#run one step for simulation,simulation timestep can be changed in ls_dog105.xml.
#@param [in] isUseRLCommand: whether use RL command to control robot. must set true if you use funtion sendCommandToRobot
def stepSimulation(isUseRLCommand):
    sim.stepSim(isUseRLCommand)

#reset simulation
def resetSimulation():
    sim.resetSim()

#send your control commands for each joint motor here,  must set isUseRLCommand be True in function stepSimulation().
#@param [in] kp: P gain  
#@param [in] kd: D gain 
#@param [in] qpos_des: target positions for motors
#@param [in] qvel_des: target velocities for motors
#@param [in] tau_ff: feedforward torques for motors
def sendCommandToRobot(kp,kd,qpos_des,qvel_des,tau_ff):
    sim.setCommand(kp,kd,qpos_des,qvel_des,tau_ff)

if __name__ == '__main__':
    initSimulation()
    counter=0 
    useRLCommand=False 
    qpos_des=[0]*12
    qpos_error=[0]*12
    kp=[200,300,100,200,300,100,200,300,100,200,300,100]
    kd=[2,3,2,2,3,2,2,3,2,2,3,2]*1
    while(True): 
        #example to run one timestep for simulation
        if useRLCommand:
            #example for joint command
            qpos = robot_state.qpos.copy() 
            qvel_des=[0]*12 
            tau_ff = [0]*12 
            delta_q=1.57*math.sin(counter/500)
            for i in range(4):
                qpos_des[0+3*i]=delta_q/2
                qpos_des[1+3*i]= delta_q/2-45/57.3
                qpos_des[2+3*i]=(random.random()*2-1)*delta_q+90/57.3 #delta_q/2+90/57.3
            for i in range(12):
                qpos_error[i]=qpos_des[i]-qpos[i]
            print("counter",counter)
            sendCommandToRobot(kp,kd,qpos_des,qvel_des,tau_ff)
            stepSimulation(True)
        else:
            stepSimulation(False)
        time.sleep(0.0001)
        #example to reset simulation
        counter+=1
        if counter%50000==0:
            print("counter",counter)
            # resetSimulation()
            counter=1
        #example to get robot state    
        robot_state= getRobotState()
            
        
