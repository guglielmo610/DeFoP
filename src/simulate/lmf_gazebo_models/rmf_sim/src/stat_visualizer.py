#!/usr/bin/env python
import rospy
import time
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState, ModelState


class StatVisualizer:
    def __init__(self):
        rospy.init_node("stat_visualizer_node")
        rospy.Subscriber('goal_topic', Pose, self.goal_cb)
        rospy.Subscriber('odom_topic', Odometry, self.odom_cb)

        self.pub_goal_x = rospy.Publisher('goal_x', Float32, queue_size=1)
        self.pub_goal_y = rospy.Publisher('goal_y', Float32, queue_size=1)
        self.pub_goal_z = rospy.Publisher('goal_z', Float32, queue_size=1)

        self.goal_x = Float32()
        self.goal_y = Float32()
        self.goal_z = Float32()

    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def goal_cb(self, msg):
        self.goal_x.data = msg.position.x
        self.goal_y.data = msg.position.y
        self.goal_z.data = msg.position.z
        self.pub_goal_x.publish(self.goal_x)
        self.pub_goal_y.publish(self.goal_y)
        self.pub_goal_z.publish(self.goal_z)


if __name__ == '__main__':
    sv = StatVisualizer()
    rospy.loginfo('Stat Visualizer ON')
    rospy.spin()
