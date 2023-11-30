from math import fmod
from re import T
import numpy as np


class GaitData:
    def __init__(self):
        self.gait_name = "STAND"
        self.gait_type = 0
        self.gait_period = 0.6
        self.phase_offset = [0]*4
        # self.gait_phase=0.0
        self.foot_phase = [0]*4
        self.switch_phase = [1.0]*4


class Gait:
    def __init__(self):
        self.cur_gait = GaitData()
        self.tar_gait = GaitData()
        self.gait_phase = 0.0
        self.foot_slip = [0]*4
        # self.gait_transition=False
        self.transition_finished = [1]*4
        self.dt = 0.001
        self.contact_target = [0]*4
        self.contact_est = [0]*4
        self.foot_state = [1]*4  # 1 means stance
        self.foot_state_next = [0]*4
        self.touchdown_early = [0]*4
        self.touchdown_late = [0]*4
        self.foot_swing_duration = [0.0]*4
        self.foot_swing_phase = [0.0]*4
        self.foot_stance_duration = [0.0]*4
        self.foot_stance_phase = [0.0]*4
        self.createGait(0)  # set target gait
        self.cur_gait.gait_name = "STAND"
        self.cur_gait.gait_name = "STAND"
        self.cur_gait.gait_type = 0
        self.cur_gait.gait_period = 0.4
        self.cur_gait.phase_offset = [0]*4
        # self.cur_gait.gait_phase=0.0
        self.cur_gait.foot_phase = [0]*4
        self.cur_gait.switch_phase = [1.0]*4
        self.delta_offset = 0.0005
        self.delta_period = 0.0005
        self.delta_switchpahse = 0.0005
        a = 1
        self.cur_phase = [0]*4
    # create gait through gait type,now only STAND,PACE,TROT,TROT_RUN gait

    def createGait(self, gait_type):
        if gait_type == 0:
            self.tar_gait.gait_name = "STAND"
            self.tar_gait.gait_type = 0
            self.tar_gait.gait_period = 0.66
            self.tar_gait.phase_offset = [0.5, 0.5, 0.5, 0.5]
            self.tar_gait.switch_phase = [1, 1, 1, 1]
        elif gait_type == 1:
            self.tar_gait.gait_name = "PACE"
            self.tar_gait.gait_type = 1
            self.tar_gait.gait_period = 0.66
            self.tar_gait.phase_offset = [0.0, 0.5, 0.0, 0.5]
            self.tar_gait.switch_phase = [0.5, 0.5, 0.5, 0.5]
        elif gait_type == 2:
            self.tar_gait.gait_name = "TROT"
            self.tar_gait.gait_type = 2
            self.tar_gait.gait_period = 0.4
            self.tar_gait.phase_offset = [0, 0.5, 0.5, 0]
            self.tar_gait.switch_phase = [0.5, 0.5, 0.5, 0.5]
        elif gait_type == 3:
            self.tar_gait.gait_name = "TROT_RUN"
            self.tar_gait.gait_type = 3
            self.tar_gait.gait_period = 0.4
            self.tar_gait.phase_offset = [0, 0.5, 0.5, 0]
            self.tar_gait.switch_phase = [0.4, 0.4, 0.4, 0.4]
        elif gait_type == 4:
            self.tar_gait.gait_name = "FIX_BASE_SWING"
            self.tar_gait.gait_type = 4
            self.tar_gait.gait_period = 0.66
            self.tar_gait.phase_offset = [0, 0, 0, 0]
            self.tar_gait.switch_phase = [0, 0, 0, 0]
        else:
            print("wrong gait type,please check")
    # check if

    def gaitTransition(self):
        # if current gait type is stand, quickly transfer to target gait type
        if self.cur_gait.gait_type == 0:
            self.gait_phase = 0.0
            for i in range(4):
                self.transition_finished[i] = True
        # current gait type is not stand,transfer to other gait type
        else:
            # self.delta_offset = 0.0005
            # self.delta_period = 0.0005
            # self.delta_switchpahse = 0.0005
            for i in range(4):
                self.transition_finished[i] = True
                # phase offset transition
                delta = self.tar_gait.phase_offset[i] - \
                    self.cur_gait.phase_offset[i]
                if np.abs(delta) > self.delta_offset:
                    self.cur_gait.phase_offset[i] = self.cur_gait.phase_offset[i] + \
                        self.delta_offset*delta/np.abs(delta)
                    self.transition_finished[i] = False
                # switch phase transition：if foot is in stance,and foot phase is less than target_switch_phase,
                # quickly change current switch_phase to target switch_phase
                if (self.cur_gait.foot_phase[i] < self.cur_gait.switch_phase[i]) and (self.cur_gait.foot_phase[i] < self.tar_gait.switch_phase[i]):
                    self.cur_gait.switch_phase[i] = self.tar_gait.switch_phase[i]
                delta = self.tar_gait.switch_phase[i] - \
                    self.cur_gait.switch_phase[i]
                if np.abs(delta) > self.delta_switchpahse:
                    self.transition_finished[i] = False
            # period transition
            delta = self.tar_gait.gait_period-self.cur_gait.gait_period
            if np.abs(delta) > self.delta_period:
                self.transition_finished[i] = False
                self.cur_gait.gait_period = self.cur_gait.gait_period + \
                    self.delta_period*delta/np.abs(delta)

        # if tansition finished ,set current gait =target gait
        if all(self.transition_finished):
            self.cur_gait.gait_type = self.tar_gait.gait_type
            self.cur_gait.gait_name = self.tar_gait.gait_name
            self.cur_gait.gait_period = self.tar_gait.gait_period
            self.cur_gait.switch_phase = self.tar_gait.switch_phase.copy()
            self.cur_gait.phase_offset = self.tar_gait.phase_offset.copy()
        else:
            print("gait is in transition,from ", self.cur_gait.gait_name,
                  " to ", self.tar_gait.gait_name)

    def updateTimeBaseGait(self):
        # if self.cur_gait.gait_type != self.tar_gait.gait_type:
        #     self.gaitTransition()
        if not self.compareTwoGait():
            self.gaitTransition()
        self.gait_phase += self.dt/self.cur_gait.gait_period
        self.gait_phase = fmod(self.gait_phase, 1.0)
        # self.cur_gait.gait_phase=self.gait_phase
        for leg in range(4):
            self.cur_gait.foot_phase[leg] = self.gait_phase +self.cur_gait.phase_offset[leg]
            self.cur_gait.foot_phase[leg] = fmod(self.cur_gait.foot_phase[leg], 1.0)
            if self.cur_gait.foot_phase[leg] > self.cur_gait.switch_phase[leg]:
                self.contact_target[leg] = 0
                self.foot_state[leg] = 0
                self.foot_state_next[leg] = 1
                self.foot_swing_phase[leg] = (self.cur_gait.foot_phase[leg]-self.cur_gait.switch_phase[leg])/(1-self.cur_gait.switch_phase[leg])
                self.foot_stance_phase[leg] = 0
            else:
                self.contact_target[leg] = 1
                self.foot_state[leg] = 1
                self.foot_state_next[leg] = 0
                self.foot_swing_phase[leg] = 0
                self.foot_stance_phase[leg] = self.cur_gait.foot_phase[leg] /self.cur_gait.switch_phase[leg]
        # print("foot state:",self.foot_state)
        # print("foot state next:",self.foot_state_next)
        # print("foot swing phase:",self.foot_swing_phase)
        # print("foot stance phase:",self.foot_stance_phase)

    def compareTwoGait(self):
        is_same = True
        if self.cur_gait.gait_type != self.tar_gait.gait_type:
            is_same = False
        if self.cur_gait.gait_name != self.tar_gait.gait_name:
            is_same = False
        if np.abs(self.cur_gait.gait_period-self.tar_gait.gait_period) > self.delta_period:
            is_same = False
        for i in range(4):
            if np.abs(self.cur_gait.phase_offset[i]-self.tar_gait.phase_offset[i] > self.delta_offset):
                is_same = False
            if np.abs(self.cur_gait.switch_phase[i]-self.tar_gait.switch_phase[i] > self.delta_switchpahse):
                is_same = False
        return is_same

    def updateEventBasedGait(self, contact_est):
        self.contact_est = contact_est.copy()
        if not self.compareTwoGait():
            self.gaitTransition()
        self.gait_phase = self.gait_phase + self.dt/self.cur_gait.gait_period
        self.gait_phase = fmod(self.gait_phase, 1.0)
        for leg in range(4):
            self.cur_gait.foot_phase[leg] = self.gait_phase + \
                self.cur_gait.phase_offset[leg]
            self.cur_gait.foot_phase[leg] = fmod(
                self.cur_gait.foot_phase[leg], 1)
            if self.cur_gait.foot_phase[leg] > self.cur_gait.switch_phase[leg]:
                self.contact_target[leg] = 0
            else:
                self.contact_target[leg] = 1
        self.touchdown_early = [0]*4
        self.touchdown_late = [0]*4
        for leg in range(4):
            # foot touchdown earlier or delay on liftoff
            if self.contact_est[leg] == 1 and self.cur_gait.foot_phase[leg] > self.cur_gait.switch_phase[leg]:
                # touch down earlier。if swing phase is bigger than half of total swing phase,transition the leg state to stance
                # 1 means target state is stance
                if self.foot_state_next[leg] == 1 and (self.cur_gait.foot_phase[leg]-self.cur_gait.switch_phase[leg] > 0.5*(1-self.cur_gait.switch_phase[leg])):
                    self.foot_state[leg] = 1
                    self.touchdown_early[leg] = 1
                # liftoff late
                else:
                    self.foot_state[leg] = 0
            # foot touchdown late or liftoff earlier
            elif self.contact_est[leg] == 0 and self.contact_target[leg] == 1:
                # foot touchdown late
                if self.foot_state_next[leg] == 1:
                    self.foot_state[leg] = 0
                    self.touchdown_late[leg] = 1
                # liftoff earlier,or slip
                else:
                    if sum(self.foot_state)-self.foot_state[leg] >= 1:
                        self.foot_state[leg] = 0
                     # make sure there are at leat two legs in stance
                    else:
                        self.foot_state[leg] = 1
                        self.foot_slip[leg] = 1
            elif self.contact_est[leg] == 1 and self.contact_target[leg] == 1:
                self.foot_state_next[leg] = 0  # 0 means swing
                self.foot_state[leg] = 1
            elif self.contact_est[leg] == 0 and self.contact_target[leg] == 0:
                self.foot_state_next[leg] = 1  # 1 means stance
                self.foot_state[leg] = 0
            else:
                print("wrong gait type")
        # modify gait phase according to self.touchdown_late and touchdown_early
        # if touch down late and stance leg not reach its limit,stop gait phase stop
        if any(self.touchdown_late) and not self.stanceLegReachLimit():
            self.gait_phase = self.gait_phase - self.dt/self.cur_gait.gait_period
            self.gait_phase = fmod(self.gait_phase+1, 1.0)
            print("touch down late")
        else:
            a = 1
        for leg in range(4):
            self.cur_gait.foot_phase[leg] = self.gait_phase + \
                self.cur_gait.phase_offset[leg]
            self.cur_gait.foot_phase[leg] = fmod(
                self.cur_gait.foot_phase[leg], 1)
            if self.touchdown_early[leg] == 1:
                a = 1
                # self.cur_gait.foot_phase[leg] = 0
            if (self.cur_gait.foot_phase[leg] > self.cur_gait.switch_phase[leg]):
                self.foot_swing_phase[leg] = (
                    self.cur_gait.foot_phase[leg]-self.cur_gait.switch_phase[leg])/(1-self.cur_gait.switch_phase[leg])
                self.foot_stance_phase[leg] = 0
            else:
                self.foot_swing_phase[leg] = 0
                self.foot_stance_phase[leg] = self.cur_gait.foot_phase[leg] / \
                    self.cur_gait.switch_phase[leg]

    def setContactEst(self, contact_est):
        for i in range(4):
            self.contact_est[i] = contact_est[i]

    def setGaitType(self, gait_type):
        if gait_type != self.tar_gait.gait_type:
            self.createGait(gait_type)

    def stanceLegReachLimit(self):  # check if any stance leg reaches its limit

        return False
