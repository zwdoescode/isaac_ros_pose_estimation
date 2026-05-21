# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import yaml


RT_DETR_MODEL_INPUT_SIZE = 640
RT_DETR_MODEL_NUM_CHANNELS = 3

REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


def _as_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _to_absolute_topic(topic):
    if not topic:
        return topic
    return topic if topic.startswith('/') else f'/{topic}'


def _load_objects(config_file):
    if not config_file:
        raise RuntimeError('object_config_file must point to a YAML file')
    if not os.path.isfile(config_file):
        raise RuntimeError(f'object_config_file does not exist: {config_file}')

    with open(config_file, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}

    objects = data.get('objects')
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f'{config_file} must contain a non-empty "objects" list')

    for index, obj in enumerate(objects):
        if 'name' not in obj:
            raise RuntimeError(f'objects[{index}] is missing required key "name"')
        if 'mesh_file_path' not in obj:
            raise RuntimeError(f'objects[{index}] is missing required key "mesh_file_path"')

    return objects


def _make_shared_rtdetr_nodes(context):
    input_width = int(LaunchConfiguration('input_width').perform(context))
    input_height = int(LaunchConfiguration('input_height').perform(context))
    rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path')
    image_topic = LaunchConfiguration('image_topic').perform(context)
    camera_info_topic = LaunchConfiguration('camera_info_topic').perform(context)
    input_to_rt_detr_ratio = input_width / RT_DETR_MODEL_INPUT_SIZE

    return [
        ComposableNode(
            name='resize_left_rt_detr_node',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ResizeNode',
            parameters=[{
                'input_width': input_width,
                'input_height': input_height,
                'output_width': RT_DETR_MODEL_INPUT_SIZE,
                'output_height': RT_DETR_MODEL_INPUT_SIZE,
                'keep_aspect_ratio': True,
                'encoding_desired': 'rgb8',
                'disable_padding': True
            }],
            remappings=[
                ('image', image_topic),
                ('camera_info', camera_info_topic),
                ('resize/image', 'rtdetr/color_image_resized'),
                ('resize/camera_info', 'rtdetr/camera_info_resized')
            ]
        ),
        ComposableNode(
            name='pad_node',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::PadNode',
            parameters=[{
                'output_image_width': RT_DETR_MODEL_INPUT_SIZE,
                'output_image_height': RT_DETR_MODEL_INPUT_SIZE,
                'padding_type': 'BOTTOM_RIGHT'
            }],
            remappings=[
                ('image', 'rtdetr/color_image_resized'),
                ('padded_image', 'rtdetr/padded_image')
            ]
        ),
        ComposableNode(
            name='image_format_node',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': RT_DETR_MODEL_INPUT_SIZE,
                'image_height': RT_DETR_MODEL_INPUT_SIZE,
            }],
            remappings=[
                ('image_raw', 'rtdetr/padded_image'),
                ('image', 'rtdetr/image_rgb')
            ]
        ),
        ComposableNode(
            name='image_to_tensor_node',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
            parameters=[{
                'scale': False,
                'tensor_name': 'image',
            }],
            remappings=[
                ('image', 'rtdetr/image_rgb'),
                ('tensor', 'rtdetr/normalized_tensor'),
            ]
        ),
        ComposableNode(
            name='interleaved_to_planar_node',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
            parameters=[{
                'input_tensor_shape': [
                    RT_DETR_MODEL_INPUT_SIZE,
                    RT_DETR_MODEL_INPUT_SIZE,
                    RT_DETR_MODEL_NUM_CHANNELS
                ]
            }],
            remappings=[
                ('interleaved_tensor', 'rtdetr/normalized_tensor'),
                ('planar_tensor', 'rtdetr/planar_tensor')
            ]
        ),
        ComposableNode(
            name='reshape_node',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
            parameters=[{
                'output_tensor_name': 'input_tensor',
                'input_tensor_shape': [
                    RT_DETR_MODEL_NUM_CHANNELS,
                    RT_DETR_MODEL_INPUT_SIZE,
                    RT_DETR_MODEL_INPUT_SIZE
                ],
                'output_tensor_shape': [
                    1,
                    RT_DETR_MODEL_NUM_CHANNELS,
                    RT_DETR_MODEL_INPUT_SIZE,
                    RT_DETR_MODEL_INPUT_SIZE
                ]
            }],
            remappings=[
                ('tensor', 'rtdetr/planar_tensor'),
                ('reshaped_tensor', 'rtdetr/reshaped_tensor')
            ],
        ),
        ComposableNode(
            name='rtdetr_preprocessor',
            package='isaac_ros_rtdetr',
            plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
            parameters=[{
                'image_width': input_width,
                'image_height': input_height,
            }],
            remappings=[
                ('encoded_tensor', 'rtdetr/reshaped_tensor')
            ]
        ),
        ComposableNode(
            name='tensor_rt',
            package='isaac_ros_tensor_rt',
            plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
            parameters=[{
                'engine_file_path': rt_detr_engine_file_path,
                'output_binding_names': ['labels', 'boxes', 'scores'],
                'output_tensor_names': ['labels', 'boxes', 'scores'],
                'input_tensor_names': ['images', 'orig_target_sizes'],
                'input_binding_names': ['images', 'orig_target_sizes'],
                'force_engine_update': False
            }]
        ),
        ComposableNode(
            name='rtdetr_decoder',
            package='isaac_ros_rtdetr',
            plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
            parameters=[{
                'confidence_threshold': float(
                    LaunchConfiguration('rtdetr_confidence_threshold').perform(context))
            }],
            remappings=[
                ('detections_output', 'detections_output')
            ]
        )
    ], int(input_width / input_to_rt_detr_ratio), int(input_height / input_to_rt_detr_ratio)


def _make_foundationpose_branch(context, obj, rtdetr_mask_width, rtdetr_mask_height):
    name = obj['name']
    namespace = obj.get('namespace', name)
    mesh_file_path = obj['mesh_file_path']
    desired_class_id = str(obj.get('desired_class_id', ''))
    detection_rank = int(obj.get('detection_rank', 0))
    segmentation_topic = obj.get('segmentation_topic', 'segmentation')
    output_topic = obj.get('output_topic', 'output')
    pose_matrix_output_topic = obj.get('pose_matrix_output_topic', 'pose_matrix_output')
    tf_frame_name = obj.get('tf_frame_name', f'fp_{name}')

    input_width = int(LaunchConfiguration('input_width').perform(context))
    input_height = int(LaunchConfiguration('input_height').perform(context))
    image_topic = _to_absolute_topic(LaunchConfiguration('image_topic').perform(context))
    depth_topic = _to_absolute_topic(LaunchConfiguration('depth_topic').perform(context))
    camera_info_topic = _to_absolute_topic(
        LaunchConfiguration('camera_info_topic').perform(context))
    use_rtdetr = _as_bool(LaunchConfiguration('use_rtdetr').perform(context))

    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')

    branch = []
    foundationpose_segmentation_topic = segmentation_topic

    if use_rtdetr:
        foundationpose_segmentation_topic = 'segmentation'
        branch.extend([
            ComposableNode(
                name='detection2_d_array_filter',
                namespace=namespace,
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
                parameters=[{
                    'desired_class_id': desired_class_id,
                    'detection_rank': detection_rank
                }],
                remappings=[
                    ('detection2_d_array', '/detections_output'),
                    ('detection2_d', 'detection2_d')
                ]
            ),
            ComposableNode(
                name='detection2_d_to_mask',
                namespace=namespace,
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
                parameters=[{
                    'mask_width': rtdetr_mask_width,
                    'mask_height': rtdetr_mask_height
                }],
                remappings=[
                    ('detection2_d', 'detection2_d'),
                    ('segmentation', 'rt_detr_segmentation')
                ]
            ),
            ComposableNode(
                name='resize_mask_node',
                namespace=namespace,
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'input_width': rtdetr_mask_width,
                    'input_height': rtdetr_mask_height,
                    'output_width': input_width,
                    'output_height': input_height,
                    'keep_aspect_ratio': False,
                    'encoding_desired': 'mono8',
                    'disable_padding': False
                }],
                remappings=[
                    ('image', 'rt_detr_segmentation'),
                    ('camera_info', '/rtdetr/camera_info_resized'),
                    ('resize/image', 'segmentation'),
                    ('resize/camera_info', 'camera_info_segmentation')
                ]
            ),
        ])

    branch.append(
        ComposableNode(
            name='foundationpose_node',
            namespace=namespace,
            package='isaac_ros_foundationpose',
            plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
            parameters=[{
                'mesh_file_path': mesh_file_path,
                'tf_frame_name': tf_frame_name,
                'refine_engine_file_path': refine_engine_file_path,
                'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
                'refine_input_binding_names': ['input1', 'input2'],
                'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
                'refine_output_binding_names': ['output1', 'output2'],
                'score_engine_file_path': score_engine_file_path,
                'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
                'score_input_binding_names': ['input1', 'input2'],
                'score_output_tensor_names': ['output_tensor'],
                'score_output_binding_names': ['output1'],
            }],
            remappings=[
                ('pose_estimation/depth_image', depth_topic),
                ('pose_estimation/image', image_topic),
                ('pose_estimation/camera_info', camera_info_topic),
                ('pose_estimation/segmentation', foundationpose_segmentation_topic),
                ('pose_estimation/output', output_topic),
                ('pose_estimation/pose_matrix_output', pose_matrix_output_topic)
            ]
        )
    )

    return branch


def _launch_setup(context, *args, **kwargs):
    del args, kwargs

    package_share_dir = get_package_share_directory('isaac_ros_foundationpose')
    object_config_file = LaunchConfiguration('object_config_file').perform(context)
    if not object_config_file:
        object_config_file = os.path.join(
            package_share_dir, 'config', 'multi_foundationpose_objects.yaml')

    objects = _load_objects(object_config_file)

    composable_nodes = []
    rtdetr_mask_width = int(LaunchConfiguration('input_width').perform(context))
    rtdetr_mask_height = int(LaunchConfiguration('input_height').perform(context))

    if _as_bool(LaunchConfiguration('use_rtdetr').perform(context)):
        rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path').perform(context)
        if not rt_detr_engine_file_path:
            raise RuntimeError('rt_detr_engine_file_path is required when use_rtdetr is True')
        shared_nodes, rtdetr_mask_width, rtdetr_mask_height = _make_shared_rtdetr_nodes(context)
        composable_nodes.extend(shared_nodes)

    for obj in objects:
        composable_nodes.extend(_make_foundationpose_branch(
            context, obj, rtdetr_mask_width, rtdetr_mask_height))

    container = ComposableNodeContainer(
        name=LaunchConfiguration('container_name').perform(context),
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return [container]


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'object_config_file',
            default_value='',
            description='YAML file containing the objects to estimate'),
        DeclareLaunchArgument(
            'use_rtdetr',
            default_value='True',
            description='Run one shared RT-DETR pipeline and per-object detection filters'),
        DeclareLaunchArgument(
            'image_topic',
            default_value='image_rect',
            description='RGB image topic used by RT-DETR and FoundationPose'),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='depth',
            description='Depth image topic used by FoundationPose'),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='camera_info_rect',
            description='CameraInfo topic used by RT-DETR and FoundationPose'),
        DeclareLaunchArgument(
            'input_width',
            default_value='640',
            description='Input RGB/depth image width'),
        DeclareLaunchArgument(
            'input_height',
            default_value='480',
            description='Input RGB/depth image height'),
        DeclareLaunchArgument(
            'rt_detr_engine_file_path',
            default_value='',
            description='Absolute path to the RT-DETR TensorRT engine'),
        DeclareLaunchArgument(
            'rtdetr_confidence_threshold',
            default_value='0.5',
            description='RT-DETR confidence threshold'),
        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_PATH,
            description='Absolute path to the FoundationPose refine TensorRT engine'),
        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_PATH,
            description='Absolute path to the FoundationPose score TensorRT engine'),
        DeclareLaunchArgument(
            'container_name',
            default_value='multi_foundationpose_container',
            description='Composable node container name'),
        OpaqueFunction(function=_launch_setup),
    ])
