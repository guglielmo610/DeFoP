cmake_minimum_required(VERSION 3.0.1)
project(lmf_planner_srvs_msgs)
find_package(catkin REQUIRED COMPONENTS
std_msgs
message_generation)

add_service_files(
  DIRECTORY srv
  FILES
  set_goal_dir.srv
)

#add_message_files(
#  DIRECTORY msg
#  FILES
#  RobotStatus.msg
#)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)