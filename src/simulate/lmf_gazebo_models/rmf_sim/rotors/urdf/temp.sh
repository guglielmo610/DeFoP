rosrun xacro xacro --inorder delta.gazebo mav_name:=delta namespace:=delta enable_logging:=false enable_ground_truth:=true enable_mavlink_interface:=false log_file:=delta wait_to_record_bag:=false > contact_robot.urdf
gz sdf -p contact_robot.urdf > contact_robot.sdf
