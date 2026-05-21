#!/usr/bin/env python3

import argparse
from array import array
from copy import deepcopy

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class DopeRgbdRepublisher(Node):
    def __init__(self, args):
        super().__init__('dope_rgbd_republisher')
        self._args = args
        self._camera_info = None
        self._depth_bytes = (
            array('f', [args.depth_m]) * (args.width * args.height)).tobytes()

        self._image_pub = self.create_publisher(Image, args.out_image_topic, 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, args.out_camera_info_topic, 10)
        self._depth_pub = self.create_publisher(Image, args.out_depth_topic, 10)

        self.create_subscription(
            CameraInfo, args.in_camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, args.in_image_topic, self._image_callback, qos_profile_sensor_data)

        self.get_logger().info(
            f'Republishing {args.in_image_topic} as {args.width}x{args.height} rgb8 on '
            f'{args.out_image_topic}, with {args.out_depth_topic}={args.depth_m}m')

    def _camera_info_callback(self, camera_info: CameraInfo):
        self._camera_info = camera_info

    def _image_callback(self, image: Image):
        if self._camera_info is None:
            self.get_logger().warn('Waiting for camera_info before publishing RGB-D frame',
                                   throttle_duration_sec=5.0)
            return

        rgb = self._resize_to_rgb8(image)
        header = image.header

        out_image = Image()
        out_image.header = header
        out_image.height = self._args.height
        out_image.width = self._args.width
        out_image.encoding = 'rgb8'
        out_image.is_bigendian = 0
        out_image.step = self._args.width * 3
        out_image.data = rgb.tobytes()

        out_depth = Image()
        out_depth.header = header
        out_depth.height = self._args.height
        out_depth.width = self._args.width
        out_depth.encoding = '32FC1'
        out_depth.is_bigendian = 0
        out_depth.step = self._args.width * 4
        out_depth.data = self._depth_bytes

        out_camera_info = self._scaled_camera_info(self._camera_info, header)

        self._camera_info_pub.publish(out_camera_info)
        self._image_pub.publish(out_image)
        self._depth_pub.publish(out_depth)

    def _resize_to_rgb8(self, image: Image):
        if image.encoding not in ('bgr8', 'rgb8'):
            raise RuntimeError(f'Unsupported image encoding: {image.encoding}')

        source = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, image.step // image.width)
        resized = cv2.resize(
            source[:, :, :3], (self._args.width, self._args.height),
            interpolation=cv2.INTER_AREA)
        if image.encoding == 'bgr8':
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return resized

    def _scaled_camera_info(self, camera_info: CameraInfo, header):
        scaled = deepcopy(camera_info)
        scale_x = self._args.width / camera_info.width
        scale_y = self._args.height / camera_info.height

        scaled.header = header
        scaled.width = self._args.width
        scaled.height = self._args.height
        scaled.k[0] *= scale_x
        scaled.k[2] *= scale_x
        scaled.k[4] *= scale_y
        scaled.k[5] *= scale_y
        scaled.p[0] *= scale_x
        scaled.p[2] *= scale_x
        scaled.p[5] *= scale_y
        scaled.p[6] *= scale_y
        return scaled


def main():
    parser = argparse.ArgumentParser(
        description='Republish the Isaac ROS DOPE RGB-only quickstart bag as a 640x480 RGB-D stream.')
    parser.add_argument('--in-image-topic', default='/image_rect')
    parser.add_argument('--in-camera-info-topic', default='/camera_info_rect')
    parser.add_argument('--out-image-topic', default='/fp/image_rect')
    parser.add_argument('--out-camera-info-topic', default='/fp/camera_info_rect')
    parser.add_argument('--out-depth-topic', default='/fp/depth')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--depth-m', type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = DopeRgbdRepublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
