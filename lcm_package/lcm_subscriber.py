import lcm
from std_msgs.msg import Float32MultiArray
#******************** import lcm message files here
from leg_control_command_lcmt import leg_control_command_lcmt
from leg_control_data_lcmt import leg_control_data_lcmt
from localization_lcmt import localization_lcmt

import time
        
class LcmInterface():
    def __init__(self):
        self.lc=lcm.LCM()
        #s***************et variables for lcm message data
        self.leg_control_command=leg_control_command_lcmt()
        self.leg_control_data=leg_control_data_lcmt()
        self.global_to_robot=localization_lcmt()
        self.subscribeLCM()

    #************** subscribe LCM data
    def subscribeLCM(self):
        self.subscription1=self.lc.subscribe("leg_control_command",self.leg_control_command_callback)
        self.subscription1.set_queue_capacity(100)# store only one data
        self.subscription3=self.lc.subscribe("leg_control_data",self.leg_control_data_callback)
        self.subscription3.set_queue_capacity(100)# store only one data
        self.subscription4=self.lc.subscribe("global_to_robot",self.global_to_robot_callback)
        self.subscription4.set_queue_capacity(100)# store only one data
    #***************register callback function
    def leg_control_command_callback(self,channel,data):
        self.leg_control_command = leg_control_command_lcmt.decode(data)
    def leg_control_data_callback(self,channel,data):
        self.leg_control_data = leg_control_data_lcmt.decode(data)
    def global_to_robot_callback(self,channel,data):
        self.global_to_robot = localization_lcmt.decode(data)   
    #***************set data to be published



# if __name__=='__main__':
#     print("start lcm test ")
#     sim.subscribeLCM() #订阅
#     #
#     while True:
#         sart_time=time.time()
#         # sim.lc.handle()     #更新
#         sim.lc.handle_timeout(10)      
#         end_time=time.time()
#         while (end_time-sart_time<0.0005):
#             end_time=time.time()
#         # print(end_time-sart_time)
