echo Do you want to reset joints?
echo 0: Yes
echo 1: No

read choice

if [ $choice -eq 0 ]
then
  rostopic pub -1 /robot/joint1_position_controller/command std_msgs/Float64 "data: 0"
  rostopic pub -1 /robot/joint2_position_controller/command std_msgs/Float64 "data: 0"
  rostopic pub -1 /robot/joint3_position_controller/command std_msgs/Float64 "data: 0"
  rostopic pub -1 /robot/joint4_position_controller/command std_msgs/Float64 "data: 0"
else
  echo joint 1 rad?

  read joint_1_rad

  echo joint 2 rad?

  read joint_2_rad

  echo joint 3 rad?

  read joint_3_rad

  echo joint 4 rad?

  read joint_4_rad

  rostopic pub -1 /robot/joint1_position_controller/command std_msgs/Float64 "data: ${joint_1_rad}"
  rostopic pub -1 /robot/joint2_position_controller/command std_msgs/Float64 "data: ${joint_2_rad}"
  rostopic pub -1 /robot/joint3_position_controller/command std_msgs/Float64 "data: ${joint_3_rad}"
  rostopic pub -1 /robot/joint4_position_controller/command std_msgs/Float64 "data: ${joint_4_rad}"

  echo "j1_${joint_1_rad}_j2_${joint_2_rad}_j3_${joint_3_rad}_j4_${joint_4_rad}" >> joins.txt


fi
