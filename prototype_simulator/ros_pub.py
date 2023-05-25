import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState


class RosPublisher:
    def __init__(self, topic_name="robot_topic"):
        # rospy.init_node(topic_name, anonymous=True)
        self.pub = rospy.Publisher(
            topic_name, Float32MultiArray, queue_size=10)
        self.multi_data = Float32MultiArray()
        self.jointState = JointState()
        # init subscriber

    def appendData(self, data):
        self.multi_data.data.append(data)

    def clearData(self):
        self.multi_data.data.clear()

    def publishData(self):
        self.pub.publish(self.multi_data)
        self.clearData()


class JointInfoPub:
    def __init__(self, topic_name="joint_State"):
        self.jointState = JointState()
        self.pub = rospy.Publisher(topic_name, JointState, queue_size=1)
        self.jointState.name = ["joint0", "joint1", "joint2", "joint3", "joint4",
                                "joint5", "joint6", "joint7", "joint8", "joint9", "joint10", "joint11"]
        self.seq = 0

    def appendData(self, q, v, tau, time):
        assert (len(q) == 12, "dimension of joint position must be 12!")
        assert (len(v) == 12, "dimension of joint velocity must be 12!")
        assert (len(tau) == 12, "dimension of joint torque must be 12!")
        self.jointState.header.frame_id = "mujoco_time"
        rostime = self.jointState.header.stamp.from_sec(time)
        self.jointState.header.stamp.secs = rostime.secs
        self.jointState.header.stamp.nsecs = rostime.nsecs
        self.jointState.position = q
        self.jointState.velocity = v
        self.jointState.effort = tau

    def publishData(self):
        self.pub.publish(self.jointState)


class BodyInfoPub:
    def __init__(self, topic_name="body_cheater_State"):
        self.body = JointState()
        self.pub = rospy.Publisher(topic_name, JointState, queue_size=1)
        self.body.name = ["px", "py", "pz", "rx", "ry", "rz"]
        self.seq = 0

    def appendData(self, q, v, time):  # q:x y z r p y(euler zyx)
        assert (len(q) == 6, "dimension of joint position must be 6!")
        assert (len(v) == 6, "dimension of joint velocity must be 6!")
        self.body.header.frame_id = "mujoco_time"
        rostime = self.body.header.stamp.from_sec(time)
        self.body.header.stamp.secs = rostime.secs
        self.body.header.stamp.nsecs = rostime.nsecs
        self.body.position = q
        self.body.velocity = v

    def publishData(self):
        self.pub.publish(self.body)
