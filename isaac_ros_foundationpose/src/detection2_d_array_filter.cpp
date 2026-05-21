// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

/*
ROS 2 node that selects a single object from a vision_msgs::msg::Detection2DArray
based on desired class ID and confidence
*/
class Detection2DArrayFilter : public rclcpp::Node
{
public:
  explicit Detection2DArrayFilter(const rclcpp::NodeOptions & options)
  : Node("detection2_d_to_mask", options),
    input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
    output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
    desired_class_id_(declare_parameter<std::string>("desired_class_id", "")),
    detection_rank_(
      static_cast<size_t>(std::max<int64_t>(0, declare_parameter<int>("detection_rank", 0)))),
    detection2_d_array_sub_{create_subscription<vision_msgs::msg::Detection2DArray>(
        "detection2_d_array", input_qos_,
        std::bind(&Detection2DArrayFilter::boundingBoxArrayCallback, this, std::placeholders::_1))},
    detection2_d_pub_{create_publisher<vision_msgs::msg::Detection2D>("detection2_d", output_qos_)}
  {}

  void boundingBoxArrayCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    std::vector<std::pair<float, vision_msgs::msg::Detection2D>> ranked_detections;
    for (const auto & detection : msg->detections) {
      float best_score = 0.0F;
      for (const auto & result : detection.results) {
        if (result.hypothesis.score > best_score && (desired_class_id_.empty() ||
          desired_class_id_ == result.hypothesis.class_id))
        {
          best_score = result.hypothesis.score;
        }
      }
      if (best_score > 0.0F) {
        ranked_detections.emplace_back(best_score, detection);
      }
    }

    if (ranked_detections.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "No detection found with non-zero confidence");
      return;
    }

    std::sort(
      ranked_detections.begin(), ranked_detections.end(),
      [](const auto & lhs, const auto & rhs) {
        return lhs.first > rhs.first;
      });

    if (detection_rank_ >= ranked_detections.size()) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "Detection rank %zu is out of range for %zu matching detections",
        detection_rank_, ranked_detections.size());
      return;
    }

    detection2_d_pub_->publish(ranked_detections[detection_rank_].second);
  }

private:
  // QOS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  std::string desired_class_id_;
  size_t detection_rank_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection2_d_array_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr detection2_d_pub_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::Detection2DArrayFilter)
