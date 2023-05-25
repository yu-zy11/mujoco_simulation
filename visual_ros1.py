import rospy
from std_msgs.msg import Float32MultiArray

class VisualRos:
    def __init__(self):
        rospy.init_node("visual_ros", anonymous=True)
        self.root_pub = rospy.Publisher("root_state", Float32MultiArray, queue_size=10)
        self.root_multi_array = Float32MultiArray()

        self.joint_pub = rospy.Publisher("joint_state", Float32MultiArray, queue_size=10)
        self.joint_multi_array = Float32MultiArray()

        # init subscriber
        self.user_data_pub = rospy.Publisher("user_input", Float32MultiArray, queue_size=10)
        self.user_input = Float32MultiArray()
        self.user_input.data = [0] * 10
        self.user_data_pub.publish(self.user_input)
        rospy.Subscriber("user_input", Float32MultiArray, self.callback)

    def append_root_data(self, data):
        self.root_multi_array.data.append(data)

    def append_joint_data(self, data):
        self.joint_multi_array.data.append(data)

    def clean_data(self):
        self.root_multi_array.data.clear()
        self.joint_multi_array.data.clear()

    def publish_data(self):
        self.root_pub.publish(self.root_multi_array)
        self.joint_pub.publish(self.joint_multi_array)
        self.clean_data()

    def callback(self, data):
        self.user_input = data
        a=1

