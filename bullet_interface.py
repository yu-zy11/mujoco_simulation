import pybullet as p
import pybullet_data
import numpy as np
import random
from scipy.spatial.transform import Rotation as sci_R


def quat_product(q1, q2):
    r1 = q1[0]
    r2 = q2[0]
    v1 = np.array([q1[1], q1[2], q1[3]])
    v2 = np.array([q2[1], q2[2], q2[3]])

    r = r1 * r2 - v1.dot(v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([r, v[0], v[1], v[2]])
    q = q / np.linalg.norm(q)
    return q


def JPL_to_hamilton_quaternion(q_jpl):
    q_hamilton = [q_jpl[3], q_jpl[0], q_jpl[1], q_jpl[2]]
    return q_hamilton

def construct_lab_terrain(uneven_terrain=False):
    plane_shape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = p.createMultiBody(0, plane_shape)
    p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0, 0, 1])

    if uneven_terrain:
        # height field
        heightPerturbationRange = 0.1
        numHeightfieldRows = 200
        numHeightfieldColumns = 50
        heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
        for j in range (int(numHeightfieldColumns/2)):
            for i in range (int(numHeightfieldRows/2)):
                height = random.uniform(0,heightPerturbationRange)
                heightfieldData[2*i+2*j*numHeightfieldRows]=height
                heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
                heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
                heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height

        terrainShape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, meshScale=[.02,.02,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
        terrain_id = p.createMultiBody(0, terrainShape)
        p.changeVisualShape(terrain_id, -1, rgbaColor=[1, 1, 1, 1])
        p.resetBasePositionAndOrientation(terrain_id, [5.0, 4.0, 0], p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        p.changeDynamics(terrain_id, -1, lateralFriction=0.1)

        box_id1  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 3, 0.05])
        p.createMultiBody(0, box_id1,  basePosition = [4.0, 4.0, 0.05])
        p.changeVisualShape(box_id1, -1, rgbaColor=[0, 0, 0, 0.8])
        p.changeDynamics(box_id1, -1, lateralFriction=1.0)
        box_id2  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 3, 0.05])
        p.createMultiBody(0, box_id2,  basePosition = [6.0, 4.0, 0.05])
        p.changeVisualShape(box_id2, -1, rgbaColor=[0, 0, 0, 0.8])
        p.changeDynamics(box_id1, -1, lateralFriction=1.0)
        box_id3  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
        p.createMultiBody(0, box_id3,  basePosition = [5.0, 6.5, 0.05])
        p.changeVisualShape(box_id3, -1, rgbaColor=[0, 0, 0, 0.8])
        p.changeDynamics(box_id1, -1, lateralFriction=1.0)
        box_id4  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
        p.createMultiBody(0, box_id4,  basePosition = [5.0, 1.5, 0.05])
        p.changeVisualShape(box_id4, -1, rgbaColor=[0, 0, 0, 0.8])
        p.changeDynamics(box_id1, -1, lateralFriction=1.0)

        # slope parameters
        offset = 2.0
        height = 0.56
        width = 2.0
        length = 1.42
        up_slope = np.deg2rad(16)
        down_slope = np.deg2rad(16)

        # stair parameters
        up_steps = 7
        up_depth = 0.30
        up_height = 0.08

        # slope_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[height/np.sin(up_slope)/2, width/2, 0.005])
        # p.createMultiBody(0, slope_shape,  basePosition = [offset+height/2/np.tan(up_slope), 0, height/2], baseOrientation = p.getQuaternionFromEuler([0, -up_slope, 0]))
        # p.changeDynamics(slope_shape, -1, lateralFriction=1.0)
        for i in range(1,up_steps+1,1):
            colSphereId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[up_depth/2, width/2, up_height/2*i])
            p.createMultiBody(0, colSphereId, basePosition=[offset+(i-0.5)*up_depth, 0, up_height/2*i])
            p.changeDynamics(colSphereId, -1, lateralFriction=1.0)

        slope_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, 0.005])
        p.createMultiBody(0, slope_shape,  basePosition = [offset+up_steps*up_depth+length/2, 0, height])
        p.changeDynamics(slope_shape, -1, lateralFriction=1.0)

        slope_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[height/np.sin(down_slope)/2, width/2, 0.005])
        p.createMultiBody(0, slope_shape,  basePosition = [offset+up_steps*up_depth+length+height/2/np.tan(down_slope), 0, height/2], baseOrientation = p.getQuaternionFromEuler([0, down_slope, 0]))
        p.changeDynamics(slope_shape, -1, lateralFriction=1.0)

    return ground_id


class BulletInterface:
    def __init__(self, settings: dict, **kwargs):
        self.robot = settings['enable_robot']
        self.cycle_time = settings['cycle_time']
        self.path = settings['path']
        self.is_display = settings['is_display']
        self.motor_id_list = range(12)
        self.foot_link_ids = [2, 5, 8, 11]
        self.init_base_position = [0, 0, 0.4]
        self.init_joint_position = [0.] * 12
        self.offset = [0.] * 12
        self.foot_radius = 0.
        self.imu_data = [0.0] * 10
        self.leg_data = [0.0] * 36
        self.foot_data = [0.0] * 4
        self.applied_torque = [0.0] * 12
        self.ground_truth = dict(
            body_position=[0.] * 3,
            body_quaternion=[0., 0., 0., 1.],
            body_linear_velocity=[0.] * 3,
            body_angular_velocity=[0.] * 3,
            body_linear_acceleration=[0.] * 3
        )
        self.last_root_velocity = [0.0] * 3
        # TODO: read from json file
        if self.robot == 'et2':
            self.init_joint_position = [-0.9, -1.2, 2.4, 0.9, -1.2, 2.4, -0.7, 0.615, -1.815, 0.7, 0.615, -1.815]
        if self.robot == 'et1_beta':
            self.init_joint_position = [-0.9, -1.2, 2.4, 0.9, -1.2, 2.4, -0.7, 0.615, -1.815, 0.7, 0.615, -1.815]
        if self.robot == 'px2_et1':
            self.init_joint_position = [-0.9, -1.2, 2.4, 0.9, -1.2, 2.4, -0.7, 0.615, -1.815, 0.7, 0.615, -1.815]
        if self.robot == 'px2_ot':
            self.init_joint_position = [-0.7, 1.2, -2.4, 0.7, 1.2, -2.4, -0.7, 0.615, -1.815, 0.7, 0.615, -1.815]
        elif self.robot == 'aliengo':
            self.init_joint_position = [-0.4, 1.8, -2.775, 0.4, 1.8, -2.775, -0.4, 1.8, -2.775, 0.4, 1.8, -2.775]
        elif self.robot == 'a1':
            self.init_joint_position = [-0.4, 1.8, -2.775, 0.4, 1.8, -2.775, -0.4, 1.8, -2.775, 0.4, 1.8, -2.775]
        else:
            pass

        self.attractor_active = False
        self.attractor_kp_lin = 1000
        self.attractor_kd_lin = 200
        self.attractor_kp_ang = 100
        self.attractor_kd_ang = 20

        self.reset()
        self.reset_robot()

    def get_states(self):
        # get base states from the simulator
        root_velocity = p.getBaseVelocity(self.robot_id)
        self.root_position, self.root_quaternion = p.getBasePositionAndOrientation(self.robot_id)
        root_euler = p.getEulerFromQuaternion(self.root_quaternion)
        # rotation matrix from world to body
        invert_transform = p.invertTransform(self.root_position, self.root_quaternion)
        rotation_matrix = p.getMatrixFromQuaternion(invert_transform[1])
        root_rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        # compute imu acceleration and angular velocity
        acc_diff = [(root_velocity[0][i] - self.last_root_velocity[i])
                    / self.cycle_time for i in range(3)]
        acc_diff[2] += 9.8
        accelerometer = root_rotation_matrix @ np.array(acc_diff).reshape(3, 1)
        gyroscope = root_rotation_matrix @ np.array(root_velocity[1][0:3]).reshape(3, 1)

        # update IMU data
        self.imu_data[0:3] = accelerometer.reshape(1, 3).tolist()[0]
        self.imu_data[3:7] = self.root_quaternion
        self.imu_data[7:10] = gyroscope.reshape(1, 3).tolist()[0]

        root_velocity_in_body = root_rotation_matrix @ np.array(root_velocity[0][0:3]).reshape(3, 1)

        # add random noise
        use_noise = True  # TODO: read from config
        if use_noise:
            self.imu_data[0:3] += np.random.uniform(-0.005, 0.005, 3)
            self.imu_data[7:10] += np.random.uniform(-0.005, 0.005, 3)
            # quaternion
            omega_noise = np.random.uniform(-0.003, 0.003, 3)
            ang = np.linalg.norm(omega_noise)
            if ang > 0.0:
                axis = omega_noise / ang
            else:
                axis = np.array([1, 0, 0])
            ee = np.sin(ang / 2) * axis
            quatD = np.array([np.cos(ang / 2), ee[0], ee[1], ee[2]])
            quatNew = quat_product(quatD, np.array(self.imu_data[3:7]))
            self.imu_data[3:7] = quatNew
            self.imu_data = [self.imu_data[i].item() for i in range(10)]

        # convert JPL convention to Hamilton convention
        self.imu_data[3:7] = JPL_to_hamilton_quaternion(self.imu_data[3:7])

        # ground truth
        body_position = list(self.root_position)
        body_position[2] -= self.foot_radius
        self.ground_truth['body_position'] = body_position
        self.ground_truth['body_orientation'] = JPL_to_hamilton_quaternion(self.root_quaternion)
        self.ground_truth['body_linear_velocity'] = root_velocity_in_body.reshape(1, 3).tolist()[0]
        self.ground_truth['body_angular_velocity'] = gyroscope.reshape(1, 3).tolist()[0]
        self.ground_truth['body_linear_acceleration'] = accelerometer.reshape(1, 3).tolist()[0]

        # get joint states from the simulator
        joint_states = p.getJointStates(self.robot_id, self.motor_id_list)
        self.leg_data[0:12] = [joint_states[i][0] + self.offset[i] for i in range(12)]
        self.leg_data[12:24] = [joint_states[i][1] for i in range(12)]
        # the jointReactionForces and appliedJointMotorTorque terms from the getJointStates interface
        # is not reasonable, use the last applied torque command instead
        self.leg_data[24:36] = self.applied_torque

        # get foot states from the simulator
        foot_link_ids = self.foot_link_ids
        for i, foot_id in enumerate(foot_link_ids):
            foot_link_contacts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=foot_id)
            contact_force = np.zeros(3)
            for foot_link_contact in foot_link_contacts:
                contact_force += foot_link_contact[9] * np.array(foot_link_contact[7]) + \
                                 foot_link_contact[10] * np.array(foot_link_contact[11]) + \
                                 foot_link_contact[12] * np.array(foot_link_contact[13])
            self.foot_data[i] = np.linalg.norm(contact_force)

        self.last_root_velocity = root_velocity[0][0:3]

        return self.imu_data, self.leg_data, self.foot_data, self.ground_truth

    def run(self):
        p.stepSimulation()

    def set_forces(self, force):
        self.applied_torque = force
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.motor_id_list,
                                    controlMode=p.TORQUE_CONTROL, forces=force)
    def read_sim_speed(self):
        return p.readUserDebugParameter(self.sim_speed)

    def reset(self):
        # initialize pybullet
        if self.is_display:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setTimeStep(self.cycle_time)
        p.setGravity(0, 0, -9.8)
        p.resetDebugVisualizerCamera(0.2, 45, -30, [1, -1, 1])
        self.sim_speed = p.addUserDebugParameter("sim_speed", 1, 100, 1)

        # initialize terrain
        ground_id = construct_lab_terrain(uneven_terrain=True)

        p.changeDynamics(ground_id, -1, lateralFriction=1.0)  # TODO: read from config

        # initialize camera
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45.0, cameraPitch=-30.0,
                                     cameraTargetPosition=self.init_base_position)

        # initialize robot
        fixed_base = False
        if self.robot == 'et2':
            self.robot_id = p.loadURDF(self.path + "/model/px1_et2/px1_et2.urdf", self.init_base_position,
                                       useFixedBase=fixed_base)
            self.foot_radius = 0.04
        if self.robot == 'et1_beta':
            self.robot_id = p.loadURDF(self.path + "/model/et1_beta/et1_beta.urdf", self.init_base_position,
                                       useFixedBase=fixed_base)
            self.foot_radius = 0.04
        if self.robot == 'px2_et1':
            self.robot_id = p.loadURDF(self.path + "/model/px2_et1/px2_et1.urdf", self.init_base_position,
                                       useFixedBase=fixed_base)
            self.foot_radius = 0.04
        if self.robot == 'px2_ot':
            self.robot_id = p.loadURDF(self.path + "/model/px2_ot/px2_ot.urdf", self.init_base_position,
                                       useFixedBase=fixed_base)
            self.foot_radius = 0.04
        elif self.robot == 'aliengo':
            self.robot_id = p.loadURDF(self.path + "/model/aliengo/aliengo.urdf", self.init_base_position,
                                       useFixedBase=fixed_base)
            self.foot_radius = 0.0265
        elif self.robot == 'a1':
            self.robot_id = p.loadURDF(self.path + "/model/a1/a1.urdf", self.init_base_position,
                                       useFixedBase=fixed_base)
            self.foot_radius = 0.02
        else:
            pass

        # reset joint id and foot id
        joint_ids = []
        foot_ids = []
        for j in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, j)
            # print('joint info: \n', joint_info)
            if joint_info[2] == p.JOINT_REVOLUTE:
                joint_ids.append(j)
            elif joint_info[2] == p.JOINT_FIXED:
                foot_ids.append(j)
            else:
                print('Error: no available joint type.')
        if len(joint_ids) == 12:
            self.motor_id_list = joint_ids
        if len(foot_ids) == 4:
            self.foot_link_ids = foot_ids

        # enable torque sensor
        for motor_id in self.motor_id_list:
            p.enableJointForceTorqueSensor(self.robot_id, motor_id, 1)

        # reset foot friction
        p.changeDynamics(self.robot_id, self.foot_link_ids[0], lateralFriction=1.0)
        p.changeDynamics(self.robot_id, self.foot_link_ids[1], lateralFriction=1.0)
        p.changeDynamics(self.robot_id, self.foot_link_ids[2], lateralFriction=1.0)
        p.changeDynamics(self.robot_id, self.foot_link_ids[3], lateralFriction=1.0)

    def reset_robot(self, body_euler=None, joint_position=None):
        if body_euler is None:
            p.resetBasePositionAndOrientation(self.robot_id, self.init_base_position, [0, 0, 0, 1])
        else:
            p.resetBasePositionAndOrientation(self.robot_id, self.init_base_position,
                                              p.getQuaternionFromEuler(body_euler))
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        if joint_position is None:
            init_joint_position = self.init_joint_position
        else:
            init_joint_position = joint_position
        for j, motor_id in enumerate(self.motor_id_list):
            p.resetJointState(self.robot_id, motor_id, init_joint_position[j], 0.0)
            # disable force
            force = 0
            p.setJointMotorControl2(self.robot_id, motor_id, p.VELOCITY_CONTROL, force=force)

        for _ in range(200):
            p.stepSimulation

    def add_force_perturbation(self,force=[0, 3000, 0],position=[0, 0, 0]):
        p.applyExternalForce(self.robot_id, -1, force, position, p.LINK_FRAME)

    def add_velocity_perturbation(self, perturbed= [0, 0.5, 0]):
        root_velocity = p.getBaseVelocity(self.robot_id)
        rotation = sci_R.from_quat(self.root_quaternion)
        rotation_matrix_wrt_z = sci_R.from_euler("z", rotation.as_euler("xyz", degrees=True)[2], degrees=True)
        print("The Euler Angle Z", rotation.as_euler("xyz", degrees=True)[2])
        perturbed_velocity = rotation_matrix_wrt_z.as_matrix() @ perturbed   # y direction, unit m/s

        perturbed_root_linear_velocity = (root_velocity[0][0]+perturbed_velocity[0],
                                          root_velocity[0][1]+perturbed_velocity[1],
                                          root_velocity[0][2]+perturbed_velocity[2])
        p.resetBaseVelocity(self.robot_id, linearVelocity=perturbed_root_linear_velocity)

    def random_orientation_reset(self):
        yaw_angle = random.randint(-180, 180)
        rotation_wrt_z = sci_R.from_euler('z', yaw_angle, degrees=True)
        orientation_quaternion = rotation_wrt_z.as_quat()
        print("This is quaternion test", yaw_angle)
        p.resetBasePositionAndOrientation(self.robot_id, self.init_base_position, orientation_quaternion)

    def view_foot_contacts(self):
        foot_link_ids = self.foot_link_ids
        for foot_id in foot_link_ids:
            foot_link_contacts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=foot_id)
            for foot_link_contact in foot_link_contacts:
                contact_point = foot_link_contact[5]
                contact_force_end = [0, 0, 0]
                for i in range(3):
                    force = foot_link_contact[9] * foot_link_contact[7][i] + \
                            foot_link_contact[10] * foot_link_contact[11][i] + \
                            foot_link_contact[12] * foot_link_contact[13][i]
                    contact_force_end[i] = contact_point[i] + 0.003 * force
                p.addUserDebugLine(contact_point, contact_force_end, lineColorRGB=[1, 0, 0], lifeTime=0.1, lineWidth=3)

    def set_visualizer_camera(self):
        [yaw, pitch, dist] = p.getDebugVisualizerCamera()[8:11]
        p.resetDebugVisualizerCamera(dist, yaw, pitch, self.root_position)

    def apply_attractor_force(self):
        root_velocity = p.getBaseVelocity(self.robot_id)
        root_lin_vel = root_velocity[0]
        root_ang_vel = root_velocity[1]
        root_euler = p.getEulerFromQuaternion(self.root_quaternion)
        if self.attractor_active:
            force = self.attractor_kp_lin * (
                    np.array(self.init_base_position) - np.array(self.root_position))
            force += self.attractor_kd_lin * (0 - np.array(root_lin_vel))
            torque = self.attractor_kp_ang * (0 - np.array(root_euler)) + self.attractor_kd_ang * (
                    0 - np.array(root_ang_vel))
            position = [0, 0, 0]
            p.applyExternalForce(self.robot_id, -1, force.tolist(), position, p.WORLD_FRAME)
            p.applyExternalTorque(self.robot_id, -1, torque.tolist(), p.WORLD_FRAME)

    def handle_keyboard(self):
        for (key, value) in p.getKeyboardEvents().items():
            # Use 114 for key 'r'
            if key == 114 and (value & p.KEY_WAS_RELEASED):
                self.reset_robot()
            # Use 118 for key 'o', velocity perturbation
            if key == 111 and (value & p.KEY_WAS_RELEASED):
                self.add_velocity_perturbation()
            # Use 112 for key 'p', force perturbation
            if key == 112 and (value & p.KEY_WAS_RELEASED):
                self.add_force_perturbation()
            # Use 113 for key 'q', random robot orientation is set in motion
            if key == 113 and (value & p.KEY_WAS_RELEASED):
                self.random_orientation_reset()
            # Use 104 for key 'h', hold robot in air
            if key == 104 and (value & p.KEY_WAS_TRIGGERED):
                self.attractor_active ^= True
