import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray


class RosPublisher(Node):
    def __init__(self,topic_name="ros_pub"):
        self.multi_data=Float32MultiArray()
        # 初始化节点名、发布器、每0.5s回调的定时器和计数器
        rclpy.init(args=None)
        super().__init__(topic_name)
        self.pub = self.create_publisher(Float32MultiArray, topic_name,10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def append_data(self, data):
        self.multi_data.data.append(data)

    def clear_data(self):
        self.multi_data.data=[]
        a=1

    def timer_callback(self):
        '''
        定时器回调函数
        '''
        # 打印并发布字符串附加计数器值的信息
        self.clear_data()
        self.append_data(self.i)
        self.pub.publish(self.multi_data)
        print(self.multi_data.data)
        self.i += 1


def main(args=None):
    # 初始化ROS2
    # rclpy.init(args=args)

    # 创建节点
    pub_client = RosPublisher("robot_state")

    # 运行节点
    rclpy.spin(pub_client)

    # 销毁节点，退出ROS2
    pub_client.node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
