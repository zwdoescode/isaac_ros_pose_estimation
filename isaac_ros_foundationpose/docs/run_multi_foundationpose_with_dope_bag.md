# Run multi_foundationpose.launch.py with the DOPE quickstart bag

This example runs `multi_foundationpose.launch.py` with:

- Bag: `/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_dope/quickstart.bag`
- Container: `isaac_ros_dev_container`
- Launch package: `isaac_ros_foundationpose`

The DOPE quickstart bag is RGB-only. It publishes `/image_rect` and
`/camera_info_rect`, with `1920x1080 bgr8` images, but it does not publish
`/depth`. Multi-FoundationPose needs RGB, camera info, depth, and segmentation.
RT-DETR provides segmentation, and the helper script below republishes the bag
as a consistent `640x480` RGB-D stream with synthetic constant depth.

The synthetic depth is only for exercising the example pipeline with this
RGB-only bag. It is not a physically correct depth source.

## One-time container fix

If launch playback fails with a Fast-CDR symbol error like:

```text
undefined symbol: _ZN8eprosima7fastcdr3Cdr9serializeEj
```

update the ABI-related ROS packages inside the container:

```bash
isaac-ros activate
sudo apt-get update
sudo apt-get install -y \
  ros-jazzy-fastcdr \
  ros-jazzy-fastrtps \
  ros-jazzy-rosidl-typesupport-fastrtps-cpp \
  ros-jazzy-rosidl-typesupport-fastrtps-c \
  ros-jazzy-rmw-fastrtps-cpp \
  ros-jazzy-rmw-fastrtps-shared-cpp
```

## Build from source

```bash
isaac-ros activate
sudo apt-get update
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_pose_estimation/isaac_ros_foundationpose --ignore-src -y
cd ${ISAAC_ROS_WS}/ && \
   colcon build --symlink-install --packages-up-to isaac_ros_foundationpose --base-paths ${ISAAC_ROS_WS}/src/isaac_ros_pose_estimation/isaac_ros_foundationpose
source install/setup.bash
```

```bash
isaac-ros activate
sudo apt-get update
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection/isaac_ros_rtdetr --ignore-src -y
sudo apt-get install -y ros-jazzy-isaac-ros-rtdetr-models-install && \
   ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula
cd ${ISAAC_ROS_WS} && \
   colcon build --symlink-install --packages-up-to isaac_ros_rtdetr --base-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection/isaac_ros_rtdetr
source install/setup.bash
```

Verify the build
```bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 pkg prefix isaac_ros_rtdetr
ros2 pkg prefix isaac_ros_foundationpose
```

## Run

Use separate terminals.

Terminal 1: republish the DOPE bag as 640x480 RGB-D:

```bash
docker exec --user admin -it isaac_ros_dev_container bash
source /opt/ros/jazzy/setup.bash
python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_foundationpose/scripts/dope_rgbd_republisher.py \
  --width 640 \
  --height 480 \
  --depth-m 1.0
```

Terminal 2: start Multi-FoundationPose:

```bash
docker exec --user admin -it isaac_ros_dev_container bash
source /opt/ros/jazzy/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch isaac_ros_foundationpose multi_foundationpose.launch.py \
  image_topic:=/fp/image_rect \
  camera_info_topic:=/fp/camera_info_rect \
  depth_topic:=/fp/depth \
  rt_detr_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan \
  refine_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  score_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose/score_trt_engine.plan
```

Note: update `image_topic`, `camera_info_topic` and `depth_topic` accordingly based on the rosbag topics

Terminal 3: play the bag:

```bash
docker exec --user admin -it isaac_ros_dev_container bash
source /opt/ros/jazzy/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 bag play /workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_dope/quickstart.bag \
  --loop \
  --rate 0.5
```

Terminal 4: verify output:

```bash
docker exec --user admin -it isaac_ros_dev_container bash
source /opt/ros/jazzy/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 topic echo --once /soup_can/output
ros2 topic echo --once /mustard/output
```

Both `/soup_can/output` and `/mustard/output` publishes a `vision_msgs/msg/Detection3DArray` pose estimate.

## Notes

The default object config is:

```text
/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_foundationpose/config/multi_foundationpose_objects.yaml
```

It uses example mustard and soup-can meshes and blank `desired_class_id` values.
For real data, update the mesh paths and detector class IDs in that YAML.

Do not feed the original `1920x1080` DOPE image directly into the launch while
leaving the launch at its default `640x480` dimensions. FoundationPose will see
depth/image and segmentation dimensions that do not match. Use the republisher
above, or make every image, depth, camera info, and mask dimension consistent.
