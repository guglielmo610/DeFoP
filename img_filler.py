#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def fill_invalid_depth_robust(depth_image, max_distance=2.0, kernel_size=100):
    depth = depth_image.astype(np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_distance)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_valid = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1)
    large_val = max_distance + 1000.0
    depth_for_min = depth.copy()
    invalid_all = (~np.isfinite(depth)) | (depth <= 0)
    depth_for_min[invalid_all] = large_val
    min_depths = cv2.erode(depth_for_min, kernel, iterations=1)
    fill_mask = invalid_all & (dilated_valid == 1) & (min_depths < large_val)
    depth[fill_mask] = min_depths[fill_mask]

    return depth

class DepthFixNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/d435/depth/image_rect_raw",
                                    Image, self.callback, queue_size=1)
        self.pub = rospy.Publisher("/d435/depth/image_filled",
                                   Image, queue_size=1)

    def callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001  # mm / m

        fixed = fill_invalid_depth_robust(depth, max_distance=3.0, kernel_size=100)

        close_pixels = np.sum(fixed < 0.2) #modify the distance threshold
        rospy.loginfo(f"Pixels under 20 cm: {close_pixels}")

        fixed_mm = (fixed * 1000.0).astype(np.uint16)
        out_msg = self.bridge.cv2_to_imgmsg(fixed_mm, encoding="16UC1")
        # out_msg = self.bridge.cv2_to_imgmsg(fixed, encoding="32FC1")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

if __name__ == "__main__":
    rospy.init_node("depth_fill_node")
    DepthFixNode()
    rospy.loginfo("Depth fill node running...")
    rospy.spin()
