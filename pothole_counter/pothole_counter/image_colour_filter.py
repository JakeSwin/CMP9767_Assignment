#!/bin/python3

import rclpy
# Import QoS settings
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class ImageColourFilter(Node):
    def __init__(self):
        super().__init__("image_colour_filter")
        self.bridge = CvBridge()
        # Define the Best Effort QOS profile compatible with camera topic
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.image_sub = self.create_subscription(Image, 
                                                  "/limo/depth_camera_link/image_raw",
                                                  self.image_callback,
                                                  qos_profile=self.qos_profile) # Set QoS Profile
        self.publisher = self.create_publisher(Image, "/filtered_image", 10)

    def image_callback(self, data):
        self.get_logger().info("Image Message Received")
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # green_filter = cv2.inRange(hsv_image, (210, 250, 180), (218, 255, 220))
        green_filter = cv2.inRange(hsv_image, (100, 0, 0), (220, 255, 255))
        contours, hierarchy = cv2.findContours(green_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] > 300:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)
        apply_mask = cv2.bitwise_and(cv_image, cv_image, mask=green_filter)
        apply_mask = cv2.cvtColor(apply_mask, cv2.COLOR_HSV2RGB)
        result = self.bridge.cv2_to_imgmsg(apply_mask, "rgb8")
        self.publisher.publish(result)

def main(args=None):
    rclpy.init(args=args)
    image_colour_filter = ImageColourFilter()
    rclpy.spin(image_colour_filter)

    image_colour_filter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
