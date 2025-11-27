#include "acc_command_converter.h"

#include "rotors_control/parameters_ros.h"
#include "rotors_control/StateAction.h"

#include <tf/transform_broadcaster.h>

namespace rotors_control
{

  AccCommandConverterNode::AccCommandConverterNode()
      : receive_first_odom(false),
        receive_thrust_cmd(false),
        receive_goal(false),
        receive_goal_training(false),
        receive_first_goal(false),
        receive_vel_cmd(false)
  {
    ros::NodeHandle nh;

    cmd_rate_thrust_sub_ = nh.subscribe("command/rate_thrust", 1,
                                        &AccCommandConverterNode::RateThrustCallback, this);

    cmd_roll_pitch_yawrate_thrust_pub_ = nh.advertise<mav_msgs::RollPitchYawrateThrust>(
        kDefaultCommandRollPitchYawrateThrustTopic, 1);
    state_action_pub_ = nh.advertise<rotors_control::StateAction>(
        "state_action", 1);
    // cmd_twist_stamped_pub_ = nh.advertise<geometry_msgs::TwistStamped>(
    //     "cmd_velocity_stamped", 1);        

    ros::NodeHandle pnh("~");
    GetRosParameter(pnh, "use_vehicle_frame", true, &use_vehicle_frame);
    GetRosParameter(pnh, "use_yaw_stabilize", false, &use_yaw_stabilize);
    GetRosParameter(pnh, "fixed_height", false, &fixed_height); // get latest height cmd from goal topic
    GetRosParameter(pnh, "swap_yaw_rate", false, &swap_yaw_rate); // use reference.angular_rates.z to store reference yaw angle

    GetRosParameter(pnh, "Kp_x", 0.0, &Kp_x);
    GetRosParameter(pnh, "Ki_x", 0.0, &Ki_x);
    GetRosParameter(pnh, "Kd_x", 0.0, &Kd_x);
    GetRosParameter(pnh, "Kp_vel_x", 0.0, &Kp_vel_x);
    GetRosParameter(pnh, "Ki_vel_x", 0.0, &Ki_vel_x);
    GetRosParameter(pnh, "Kd_vel_x", 0.0, &Kd_vel_x);    
    GetRosParameter(pnh, "acc_x_max", 0.0, &acc_x_max);
    GetRosParameter(pnh, "alpha_x", 0.0, &alpha_x);

    GetRosParameter(pnh, "Kp_y", 0.0, &Kp_y);
    GetRosParameter(pnh, "Ki_y", 0.0, &Ki_y);
    GetRosParameter(pnh, "Kd_y", 0.0, &Kd_y);
    GetRosParameter(pnh, "Kp_vel_y", 0.0, &Kp_vel_y);
    GetRosParameter(pnh, "Ki_vel_y", 0.0, &Ki_vel_y);
    GetRosParameter(pnh, "Kd_vel_y", 0.0, &Kd_vel_y);    
    GetRosParameter(pnh, "acc_y_max", 0.0, &acc_y_max);
    GetRosParameter(pnh, "alpha_y", 0.0, &alpha_y);

    GetRosParameter(pnh, "Kp_z", 0.0, &Kp_z);
    GetRosParameter(pnh, "Ki_z", 0.0, &Ki_z);
    GetRosParameter(pnh, "Kd_z", 0.0, &Kd_z);
    GetRosParameter(pnh, "Kp_vel_z", 0.0, &Kp_vel_z);
    GetRosParameter(pnh, "Ki_vel_z", 0.0, &Ki_vel_z);
    GetRosParameter(pnh, "Kd_vel_z", 0.0, &Kd_vel_z);    
    GetRosParameter(pnh, "acc_z_max", 0.0, &acc_z_max);
    GetRosParameter(pnh, "alpha_z", 0.0, &alpha_z);

    if (!pnh.getParam("antiwindup_radius_pos", antiwindup_radius_pos)) {
        ROS_ERROR("antiwindup_radius_pos in lmf_controller is not loaded from ros parameter server");
        abort();
    }
    ROS_WARN_STREAM("antiwindup_radius_pos:" << antiwindup_radius_pos.at(0) << " "
                                            << antiwindup_radius_pos.at(1) << " "
                                            << antiwindup_radius_pos.at(2));

    if (!pnh.getParam("antiwindup_radius_vel", antiwindup_radius_vel)) {
        ROS_ERROR("antiwindup_radius_vel in lmf_controller is not loaded from ros parameter server");
        abort();
    }
    ROS_WARN_STREAM("antiwindup_radius_vel:" << antiwindup_radius_vel.at(0) << " "
                                            << antiwindup_radius_vel.at(1) << " "
                                            << antiwindup_radius_vel.at(2));

    if (!pnh.getParam("integrator_pos_max", integrator_pos_max)) {
        ROS_ERROR("integrator_pos_max in lmf_controller is not loaded from ros parameter server");
        abort();
    }
    ROS_WARN_STREAM("integrator_pos_max:" << integrator_pos_max.at(0) << " "
                                            << integrator_pos_max.at(1) << " "
                                            << integrator_pos_max.at(2));

    if (!pnh.getParam("integrator_vel_max", integrator_vel_max)) {
        ROS_ERROR("integrator_vel_max in lmf_controller is not loaded from ros parameter server");
        abort();
    }
    ROS_WARN_STREAM("integrator_vel_max:" << integrator_vel_max.at(0) << " "
                                            << integrator_vel_max.at(1) << " "
                                            << integrator_vel_max.at(2));

    GetRosParameter(pnh, "eps_explore", 0.0, &eps_explore);

    GetRosParameter(pnh, "noise_x", 0.0, &noise_x);
    GetRosParameter(pnh, "noise_y", 0.0, &noise_y);
    GetRosParameter(pnh, "noise_z", 0.0, &noise_z);
    GetRosParameter(pnh, "K_yaw", 1.8, &K_yaw);
    GetRosParameter(pnh, "yaw_rate_limit", M_PI / 4, &yaw_rate_limit);
    GetRosParameter(pnh, "mass", 1.0, &mass);
    ROS_WARN_STREAM("Mass:" << mass);

    rate_thrust_cmd.thrust.x = 0.0;
    rate_thrust_cmd.thrust.y = 0.0;
    rate_thrust_cmd.thrust.z = 0.0; // keep it on the ground
    rate_thrust_cmd.angular_rates.x = 0.0;
    rate_thrust_cmd.angular_rates.y = 0.0;
    rate_thrust_cmd.angular_rates.z = 0.0;

    gen_x = std::mt19937(rd_x());
    gen_y = std::mt19937(rd_y());
    gen_z = std::mt19937(rd_z());

    d_x = std::normal_distribution<>(0, noise_x);
    d_y = std::normal_distribution<>(0, noise_y);
    d_z = std::normal_distribution<>(0, noise_z);

    srand((unsigned)time(NULL));

    //need to know current yaw angle of the robot if acc vector is expressed in world frame
    odometry_sub_ = nh.subscribe(kDefaultOdometryTopic, 100, &AccCommandConverterNode::OdometryCallback, this);
    goal_pose_sub_ = nh.subscribe("goal", 1, &AccCommandConverterNode::GoalPoseCallback, this);
    goal_training_pose_sub_ = nh.subscribe("goal_training", 1, &AccCommandConverterNode::GoalTrainingPoseCallback, this);
    cmd_velocity_sub_ = nh.subscribe("cmd_velocity", 1, &AccCommandConverterNode::CmdVelocityCallback, this);

    reset_service_ = nh.advertiseService("pid_reset", &AccCommandConverterNode::ResetCallback, this);

    vehicle_frame_id = ros::this_node::getNamespace();
    ROS_WARN_STREAM("vehicle_frame_id:" << vehicle_frame_id);
  }

  AccCommandConverterNode::~AccCommandConverterNode() {}

  bool AccCommandConverterNode::ResetCallback(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
  {
    ROS_INFO("Reset serviced called");
    pid_x->reset();
    pid_y->reset();
    pid_z->reset();
    pid_vel_x->reset();
    pid_vel_y->reset();
    pid_vel_z->reset();
    rate_thrust_cmd.thrust.x = 0.0;
    rate_thrust_cmd.thrust.y = 0.0;
    rate_thrust_cmd.thrust.z = 0.0;
    rate_thrust_cmd.angular_rates.x = 0.0;
    rate_thrust_cmd.angular_rates.y = 0.0;
    rate_thrust_cmd.angular_rates.z = 0.0; 
    //receive_first_odom = false;
    receive_thrust_cmd = false;
    receive_goal = false;
    receive_goal_training = false;
    receive_vel_cmd = false;
    receive_first_goal = false;
    // Eigen::Vector3d robot_euler_angles;
    // odometry.getEulerAngles(&robot_euler_angles);
    // ROS_INFO_STREAM("Robot RPY:" << robot_euler_angles.transpose() * 180.0/M_PI << " deg");
    // ROS_INFO_STREAM("Robot pos W:" << odometry.position_W.transpose());
    // ROS_INFO_STREAM("Robot velocity B:" << odometry.velocity_B.transpose());
    // ROS_INFO_STREAM("Robot angular velocity B:" << odometry.angular_velocity_B.transpose());    
    return true;
  }

  void AccCommandConverterNode::RateThrustCallback(
      const mav_msgs::RateThrustPtr &rate_thrust_msg)
  {
    if (!receive_first_odom)
    {
      return;
    }
    rate_thrust_cmd = *rate_thrust_msg;
    receive_thrust_cmd = true;
    receive_goal = false;
    receive_vel_cmd = false;
    //receive_goal_training = false;
  }

  // should be in a seperate node
  void AccCommandConverterNode::GoalPoseCallback(const geometry_msgs::Pose &goal)
  {
    geometry_msgs::Pose goal_msg(goal);
    if ((goal_msg.orientation.x == 0.0) && (goal_msg.orientation.y == 0.0)
    && (goal_msg.orientation.z == 0.0) && (goal_msg.orientation.w == 0.0))
    {
      goal_msg.orientation.w = 1.0;
    }
    convertGoal2WorldFrame(goal_msg, odometry, &goal_odometry, use_vehicle_frame);
    Eigen::Vector3d goal_euler_angles;
    goal_odometry.getEulerAngles(&goal_euler_angles);
    goal_yaw = goal_euler_angles(2);
    receive_goal = true;
    receive_first_goal = true;
    receive_goal_training = false;
    receive_vel_cmd = false;
    ROS_INFO_STREAM("Received goal: pos x " << goal_msg.position.x << ",y " << goal_msg.position.y << ",z " << goal_msg.position.z
                                            << ", orientation x " << goal_msg.orientation.x << ",y " << goal_msg.orientation.y << ",z " << goal_msg.orientation.z << ",w " << goal_msg.orientation.w);
    ROS_INFO_STREAM("Odom in world: pos x " << odometry.position_W(0) << ",y " << odometry.position_W(1) << ",z " << odometry.position_W(2));
    ROS_INFO_STREAM("Goal in world: pos x " << goal_odometry.position_W(0) << ",y " << goal_odometry.position_W(1) << ",z " << goal_odometry.position_W(2));
    ROS_INFO_STREAM("Goal yaw:" << goal_yaw * 180 / M_PI << " deg");
    // Eigen::Vector3d robot_euler_angles;
    // odometry.getEulerAngles(&robot_euler_angles);
    // ROS_INFO_STREAM("Robot RPY:" << robot_euler_angles.transpose() * 180.0/M_PI << " deg");
    // ROS_INFO_STREAM("Robot pos W:" << odometry.position_W.transpose());
    // ROS_INFO_STREAM("Robot velocity B:" << odometry.velocity_B.transpose());
    // ROS_INFO_STREAM("Robot angular velocity B:" << odometry.angular_velocity_B.transpose());    
    ROS_INFO("**********");
  }

  void AccCommandConverterNode::GoalTrainingPoseCallback(const geometry_msgs::Pose &goal)
  {
    geometry_msgs::Pose goal_msg(goal);
    if ((goal_msg.orientation.x == 0.0) && (goal_msg.orientation.y == 0.0)
    && (goal_msg.orientation.z == 0.0) && (goal_msg.orientation.w == 0.0))
    {
      goal_msg.orientation.w = 1.0;
    }
    convertGoal2WorldFrame(goal_msg, odometry, &goal_training_odometry, false); // goal_training from rotors_wrapper is already in world frame
    Eigen::Vector3d goal_euler_angles;
    goal_training_odometry.getEulerAngles(&goal_euler_angles);
    goal_training_yaw = goal_euler_angles(2);
    receive_goal_training = true;
    receive_goal = false;
    receive_vel_cmd = false;
    ROS_INFO_STREAM("Received goal_training: pos x " << goal_msg.position.x << ",y " << goal_msg.position.y << ",z " << goal_msg.position.z
                                                     << ", orientation x " << goal_msg.orientation.x << ",y " << goal_msg.orientation.y << ",z " << goal_msg.orientation.z << ",w " << goal_msg.orientation.w);
    ROS_INFO_STREAM("Odom in world: pos x " << odometry.position_W(0) << ",y " << odometry.position_W(1) << ",z " << odometry.position_W(2));
    ROS_INFO_STREAM("Goal_training in world: pos x " << goal_training_odometry.position_W(0) << ",y " << goal_training_odometry.position_W(1) << ",z " << goal_training_odometry.position_W(2));
    ROS_INFO_STREAM("Goal_training yaw:" << goal_training_yaw * 180 / M_PI << " deg");
    ROS_INFO("**********");
  }

  void AccCommandConverterNode::CmdVelocityCallback(const geometry_msgs::Twist &cmd_vel)
  {
    //convertCmdVel2WorldFrame(cmd_vel, odometry, cmd_vel_W);
    cmd_vel_V = cmd_vel;
    receive_goal_training = false;
    receive_goal = false;
    receive_vel_cmd = true;

    // geometry_msgs::TwistStamped cmd_vel_stamped;
    // cmd_vel_stamped.header.stamp = ros::Time::now();
    // cmd_vel_stamped.twist = cmd_vel;
    // cmd_twist_stamped_pub_.publish(cmd_vel_stamped);    
    
    ROS_INFO_STREAM("Received cmd_vel, cmd_vel: linear x " << cmd_vel.linear.x << ",y " << cmd_vel.linear.y << ",z " << cmd_vel.linear.z
                    << ", twist x " << cmd_vel.angular.x << ",y " << cmd_vel.angular.y << ",z " << cmd_vel.angular.z);
    // Eigen::Vector3d robot_euler_angles;
    // odometry.getEulerAngles(&robot_euler_angles);
    // ROS_INFO_STREAM("Robot RPY:" << robot_euler_angles.transpose() * 180.0/M_PI << " deg");
    // ROS_INFO_STREAM("Robot pos W:" << odometry.position_W.transpose());
    // ROS_INFO_STREAM("Robot velocity B:" << odometry.velocity_B.transpose());
    // ROS_INFO_STREAM("Robot angular velocity B:" << odometry.angular_velocity_B.transpose());                    
    // ROS_INFO_STREAM("cmd_vel_W: linear x " << cmd_vel_W.linear.x << ",y " << cmd_vel_W.linear.y << ",z " << cmd_vel_W.linear.z
    //                 << ", twist x " << cmd_vel_W.twist.x << ",y " << cmd_vel_W.twist.y << ",z " << cmd_vel_W.twist.z);
    ROS_INFO("**********");    
  }

  // void AccCommandConverterNode::convertCmdVel2WorldFrame(const geometry_msgs::Twist &cmd_vel, const mav_msgs::EigenOdometry &robot_odom,
  //                                                       geometry_msgs::Twist &cmd_vel_W)
  // {
  //   if (use_vehicle_frame)
  //   {
  //     Eigen::Vector3d linear_vel_V, linear_vel_W;
  //     robot_odom.getEulerAngles(&robot_euler_angles);
  //     linear_vel_V << cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.linear.z;
  //     linear_vel_W = Eigen::AngleAxisd(robot_euler_angles(2), Eigen::Vector3d::UnitZ()) * linear_vel_V;
  //     cmd_vel_W.linear.x = linear_vel_W(0);
  //     cmd_vel_W.linear.y = linear_vel_W(1);
  //     cmd_vel_W.linear.z = linear_vel_W(2);
  //     cmd_vel_W.twist = cmd_vel.twist;
  //   }
  //   else
  //   {
  //     cmd_vel_W = cmd_vel;
  //   }
  // }

  void AccCommandConverterNode::convertGoal2WorldFrame(const geometry_msgs::Pose &goal, const mav_msgs::EigenOdometry &robot_odom,
                                                       mav_msgs::EigenOdometry *goal_in_world, bool use_vehicle_frame)
  {
    Eigen::Vector3d robot_euler_angles;
    robot_odom.getEulerAngles(&robot_euler_angles);
    Eigen::Quaterniond goal_quat_in_world;
    Eigen::Vector3d goal_pos_in_world;
    if (use_vehicle_frame)
    {
      Eigen::Quaterniond goal_quat_in_vehicle = Eigen::Quaterniond(goal.orientation.w, goal.orientation.x, goal.orientation.y, goal.orientation.z);
      goal_quat_in_vehicle = goal_quat_in_vehicle.normalized();
      Eigen::Vector3d goal_euler_angles;
      mav_msgs::getEulerAnglesFromQuaternion(goal_quat_in_vehicle, &goal_euler_angles);
      goal_quat_in_world = Eigen::AngleAxisd(goal_euler_angles(2) + robot_euler_angles(2), Eigen::Vector3d::UnitZ());
      Eigen::Vector3d goal_pos_in_vehicle;
      goal_pos_in_vehicle << goal.position.x, goal.position.y, goal.position.z;
      goal_pos_in_world = Eigen::AngleAxisd(robot_euler_angles(2), Eigen::Vector3d::UnitZ()) * goal_pos_in_vehicle + robot_odom.position_W;
    }
    else
    {
      goal_quat_in_world = Eigen::Quaterniond(goal.orientation.w, goal.orientation.x, goal.orientation.y, goal.orientation.z);
      goal_quat_in_world = goal_quat_in_world.normalized();
      goal_pos_in_world(0) = goal.position.x;
      goal_pos_in_world(1) = goal.position.y;
      goal_pos_in_world(2) = goal.position.z;
    }
    goal_in_world->position_W = goal_pos_in_world;
    goal_in_world->orientation_W_B = goal_quat_in_world;
  }

  void AccCommandConverterNode::convertGoal2VehicleFrame(const mav_msgs::EigenOdometry &goal_odom, const mav_msgs::EigenOdometry &robot_odom,
                                                         nav_msgs::Odometry *goal_in_vehicle_frame)
  {
    Eigen::Vector3d goal_euler_angles, robot_euler_angles;
    goal_odom.getEulerAngles(&goal_euler_angles);
    robot_odom.getEulerAngles(&robot_euler_angles);
    Eigen::Quaterniond quat_VG;
    quat_VG = Eigen::AngleAxisd(goal_euler_angles(2) - robot_euler_angles(2), Eigen::Vector3d::UnitZ());
    Eigen::Vector3d pos_VG = Eigen::AngleAxisd(-robot_euler_angles(2), Eigen::Vector3d::UnitZ()) * (goal_odom.position_W - robot_odom.position_W);

    goal_in_vehicle_frame->header.stamp = ros::Time::now();
    goal_in_vehicle_frame->pose.pose.position.x = pos_VG(0);
    goal_in_vehicle_frame->pose.pose.position.y = pos_VG(1);
    goal_in_vehicle_frame->pose.pose.position.z = pos_VG(2);
    goal_in_vehicle_frame->pose.pose.orientation.x = quat_VG.x();
    goal_in_vehicle_frame->pose.pose.orientation.y = quat_VG.y();
    goal_in_vehicle_frame->pose.pose.orientation.z = quat_VG.z();
    goal_in_vehicle_frame->pose.pose.orientation.w = quat_VG.w();

    goal_in_vehicle_frame->twist.twist.linear.x = -robot_odom.velocity_B(0);
    goal_in_vehicle_frame->twist.twist.linear.y = -robot_odom.velocity_B(1);
    goal_in_vehicle_frame->twist.twist.linear.z = -robot_odom.velocity_B(2);
    goal_in_vehicle_frame->twist.twist.angular.x = -robot_odom.angular_velocity_B(0);
    goal_in_vehicle_frame->twist.twist.angular.y = -robot_odom.angular_velocity_B(1);
    goal_in_vehicle_frame->twist.twist.angular.z = -robot_odom.angular_velocity_B(2);
  }

  double AccCommandConverterNode::calculateYawCtrl(double setpoint_yaw, double current_yaw)
  {
    double yaw_error = setpoint_yaw - current_yaw;

    if (std::abs(yaw_error) > M_PI)
    {
      if (yaw_error > 0.0)
      {
        while (yaw_error > M_PI)
        {
          yaw_error = yaw_error - 2.0 * M_PI;
        }
      }
      else
      {
        while (yaw_error < -M_PI)
        {
          yaw_error = yaw_error + 2.0 * M_PI;
        }
      }
    }

    double yaw_rate_cmd = K_yaw * yaw_error;

    if (yaw_rate_cmd > yaw_rate_limit)
    {
      yaw_rate_cmd = yaw_rate_limit;
    }

    if (yaw_rate_cmd < -yaw_rate_limit)
    {
      yaw_rate_cmd = -yaw_rate_limit;
    }

    return yaw_rate_cmd;    
  }

  void AccCommandConverterNode::OdometryCallback(const nav_msgs::OdometryConstPtr &odometry_msg)
  {
    ROS_INFO_ONCE("AccCommandConverter node got first odometry message.");
    if (!receive_first_odom)
    {
      static int cnt = 0;
      static ros::Time previous_time = odometry_msg->header.stamp;
      ros::Time current_time = odometry_msg->header.stamp;
      odom_dtime += (current_time - previous_time).toSec();
      previous_time = current_time;
      cnt += 1;
      if (cnt >= 51)
      {
        odom_dtime = odom_dtime / 50;
        //odom_dtime = 0.001;
        ROS_INFO_STREAM("Odom dtime:" << odom_dtime);
        frame_id = odometry_msg->header.frame_id;
        pid_x = new PID(odom_dtime, acc_x_max, -acc_x_max, Kp_x, Kd_x, Ki_x, 0.0, alpha_x, antiwindup_radius_pos.at(0), integrator_pos_max.at(0));
        pid_y = new PID(odom_dtime, acc_y_max, -acc_y_max, Kp_y, Kd_y, Ki_y, 0.0, alpha_y, antiwindup_radius_pos.at(1), integrator_pos_max.at(1));
        pid_z = new PID(odom_dtime, acc_z_max, -acc_z_max, Kp_z, Kd_z, Ki_z, 0.0, alpha_z, antiwindup_radius_pos.at(2), integrator_pos_max.at(2));
        pid_vel_x = new PID(odom_dtime, acc_x_max, -acc_x_max, Kp_vel_x, Kd_vel_x, Ki_vel_x, alpha_x, 0.0, antiwindup_radius_vel.at(0), integrator_vel_max.at(0));
        pid_vel_y = new PID(odom_dtime, acc_y_max, -acc_y_max, Kp_vel_y, Kd_vel_y, Ki_vel_y, alpha_y, 0.0, antiwindup_radius_vel.at(1), integrator_vel_max.at(1));
        pid_vel_z = new PID(odom_dtime, acc_z_max, -acc_z_max, Kp_vel_z, Kd_vel_z, Ki_vel_z, alpha_z, 0.0, antiwindup_radius_vel.at(2), integrator_vel_max.at(2));

        receive_first_odom = true;
      }
      else
      {
        return;
      }
    }
    mav_msgs::eigenOdometryFromMsg(*odometry_msg, &odometry);
    nav_msgs::Odometry goal_in_approriate_frame;
    Eigen::Vector3d current_rpy;
    odometry.getEulerAngles(&current_rpy);

    if (receive_goal)
    {
      mav_msgs::RateThrust rate_thrust_cmd_tmp;
      // calculate goal in vehicle frame
      if (use_vehicle_frame)
      {
        convertGoal2VehicleFrame(goal_odometry, odometry, &goal_in_approriate_frame);
        rate_thrust_cmd_tmp.thrust.x = pid_x->calculate(0.0, -goal_in_approriate_frame.pose.pose.position.x);
        rate_thrust_cmd_tmp.thrust.y = pid_y->calculate(0.0, -goal_in_approriate_frame.pose.pose.position.y);
        rate_thrust_cmd_tmp.thrust.z = pid_z->calculate(0.0, -goal_in_approriate_frame.pose.pose.position.z);
      }
      else
      {
        msgOdometryFromEigen(goal_odometry, &goal_in_approriate_frame);
        rate_thrust_cmd_tmp.thrust.x = pid_x->calculate(goal_odometry.position_W(0), odometry.position_W(0));
        rate_thrust_cmd_tmp.thrust.y = pid_y->calculate(goal_odometry.position_W(1), odometry.position_W(1));
        rate_thrust_cmd_tmp.thrust.z = pid_z->calculate(goal_odometry.position_W(2), odometry.position_W(2));
      }

      float random_num = (float)rand() / RAND_MAX;
      if (random_num < eps_explore)
      {
        double nx, ny, nz;
        nx = d_x(gen_x);
        ny = d_y(gen_y);
        nz = d_z(gen_z);
        //std::cout << "Noise:" << nx << "," << ny << "," << nz << "," << random_num << std::endl;
        rate_thrust_cmd_tmp.thrust.x += nx;
        rate_thrust_cmd_tmp.thrust.y += ny;
        rate_thrust_cmd_tmp.thrust.z += nz;
      }
      rate_thrust_cmd_tmp.angular_rates.x = 0.0;
      rate_thrust_cmd_tmp.angular_rates.y = 0.0;
      rate_thrust_cmd_tmp.angular_rates.z = 0.0;
      rate_thrust_cmd = rate_thrust_cmd_tmp;
    }
    else if (receive_goal_training) // receive goal but don't execute
    {
      // calculate goal in vehicle frame
      // if (use_vehicle_frame)
      // {
      //   convertGoal2VehicleFrame(goal_training_odometry, odometry, &goal_in_approriate_frame);
      // }      
      // else
      // {
        msgOdometryFromEigen(goal_training_odometry, &goal_in_approriate_frame); // already in world frame
      //}
    }
    else if (receive_vel_cmd)
    {
      if (use_vehicle_frame)
      {
        // convert odom in W to V
        Eigen::Vector3d linear_vel_W, linear_vel_V;
        linear_vel_W = odometry.getVelocityWorld();
        linear_vel_V = Eigen::AngleAxisd(-current_rpy(2), Eigen::Vector3d::UnitZ()) * linear_vel_W;
        //ROS_INFO_STREAM("linear_vel_V: " << linear_vel_V.transpose());
        //ROS_INFO_STREAM("linear_vel_W: " << linear_vel_W.transpose());
        rate_thrust_cmd.thrust.x = pid_vel_x->calculate(cmd_vel_V.linear.x, linear_vel_V(0));
        rate_thrust_cmd.thrust.y = pid_vel_y->calculate(cmd_vel_V.linear.y, linear_vel_V(1));
        if (!fixed_height)
        {
          rate_thrust_cmd.thrust.z = pid_vel_z->calculate(cmd_vel_V.linear.z, linear_vel_V(2));
        }
      }
      else
      {
        /* not implemented yet */
      }      
    }

    if (fixed_height && (receive_thrust_cmd || receive_vel_cmd) && receive_first_goal)
    {
      // modify z thrust to keep the same height
      rate_thrust_cmd.thrust.z = pid_z->calculate(goal_odometry.position_W(2), odometry.position_W(2));
    }
    mav_msgs::RateThrust reference = rate_thrust_cmd;
    mav_msgs::RollPitchYawrateThrustPtr rpyrate_thrust_cmd(new mav_msgs::RollPitchYawrateThrust);

    // double current_roll = current_rpy(0);
    // double current_pitch = current_rpy(1);
    double current_yaw = (use_vehicle_frame) ? 0.0 : current_rpy(2);
    Eigen::Vector3d thrust_sp;
    thrust_sp << reference.thrust.x, reference.thrust.y, reference.thrust.z;
    thrust_sp = (thrust_sp + Eigen::Vector3d(0, 0, kGravity)) * mass;
    Eigen::Vector3d thrust_norm = thrust_sp.normalized();

    double cphi_stheta = thrust_norm(0) * cos(current_yaw) + thrust_norm(1) * sin(current_yaw);
    double sphi = thrust_norm(0) * sin(current_yaw) - thrust_norm(1) * cos(current_yaw);
    double cphi_ctheta = thrust_norm(2);
    if (cphi_ctheta != 0)
    {
      rpyrate_thrust_cmd->pitch = atan2(cphi_stheta, cphi_ctheta);
      rpyrate_thrust_cmd->roll = atan2(sphi, sqrt(cphi_stheta * cphi_stheta + cphi_ctheta * cphi_ctheta));
    }
    else
    {
      rpyrate_thrust_cmd->pitch = 0;
      rpyrate_thrust_cmd->roll = 0;
    }

    // YAW ctrl
    if (receive_goal)
    {
      Eigen::Vector3d goal_euler_angles;
      double yaw_rate_cmd = calculateYawCtrl(goal_yaw, current_rpy(2));
      rpyrate_thrust_cmd->yaw_rate = yaw_rate_cmd;
    }
    else if (receive_vel_cmd)
    {
      if (swap_yaw_rate) // use cmd_vel.twist.z to store reference yaw angle 
      {
        double yaw_rate_cmd = calculateYawCtrl(cmd_vel_V.angular.z, current_rpy(2));
        rpyrate_thrust_cmd->yaw_rate = yaw_rate_cmd;
      }
      else
      {
        rpyrate_thrust_cmd->yaw_rate = cmd_vel_V.angular.z;
      }      
    }
    else if (receive_thrust_cmd)
    {
      if (use_yaw_stabilize)
      {
        double yaw_rate_cmd = calculateYawCtrl(0.0, current_rpy(2));
        rpyrate_thrust_cmd->yaw_rate = yaw_rate_cmd;
      }
      else
      {
        if (swap_yaw_rate) // use reference.angular_rates.z to store reference yaw angle 
        {
          double yaw_rate_cmd = calculateYawCtrl(reference.angular_rates.z, current_rpy(2));
          rpyrate_thrust_cmd->yaw_rate = yaw_rate_cmd;
        }
        else
        {
          rpyrate_thrust_cmd->yaw_rate = reference.angular_rates.z;
        }
      }
    }

    rpyrate_thrust_cmd->thrust.x = 0;
    rpyrate_thrust_cmd->thrust.y = 0;
    // total thrust is the norm of thrust vector
    rpyrate_thrust_cmd->thrust.z = thrust_sp.norm();

    // OR cross-product of thrust vector and current z_B axis
    //rpyrate_thrust_cmd->thrust.z = thrust_sp(0)*(cos(current_roll)*sin(current_pitch)*cos(current_yaw) + sin(current_roll)*sin(current_yaw))
    //                           + thrust_sp(1)*(cos(current_roll)*sin(current_pitch)*sin(current_yaw) - sin(current_roll)*cos(current_yaw))
    //                           + thrust_sp(2)*cos(current_roll)*cos(current_pitch);

    rpyrate_thrust_cmd->header.frame_id = frame_id;
    rpyrate_thrust_cmd->header.stamp = ros::Time::now();

    cmd_roll_pitch_yawrate_thrust_pub_.publish(rpyrate_thrust_cmd);

    static mav_msgs::RateThrust previous_action = reference;
    static nav_msgs::Odometry previous_robot_odom = *odometry_msg;
    static nav_msgs::Odometry previous_goal_odom = goal_in_approriate_frame;
    static bool first_msg = true;

    if (first_msg)
    {
      first_msg = false;
    }
    else
    {
      StateAction state_action_msg;
      state_action_msg.header.stamp = ros::Time::now();//previous_robot_odom.header.stamp;
      state_action_msg.robot_odom = previous_robot_odom;
      state_action_msg.goal_odom = previous_goal_odom;
      state_action_msg.action = previous_action;
      state_action_msg.next_goal_odom = goal_in_approriate_frame;
      state_action_msg.use_vehicle_frame = use_vehicle_frame;
      state_action_msg.collide = false;
      state_action_pub_.publish(state_action_msg);

      previous_action = reference;
      previous_robot_odom = *odometry_msg;
      previous_goal_odom = goal_in_approriate_frame;
    }

    // publish tf for vehicle frame
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin( tf::Vector3(odometry.position_W(0), odometry.position_W(1), odometry.position_W(2)) );
    tf::Quaternion q;
    q.setRPY(0, 0, current_rpy(2));
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odometry_msg->header.stamp, "world", vehicle_frame_id));    
  }

} // namespace rotors_control

int main(int argc, char **argv)
{
  ros::init(argc, argv, "acc_command_converter");

  rotors_control::AccCommandConverterNode acc_command_converter_node;

  ros::spin();

  return 0;
}