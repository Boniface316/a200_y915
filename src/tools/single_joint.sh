echo Joint number?

read joint

echo rad?

read rad

rostopic pub -1 /robot/joint${joint}_position_controller/command std_msgs/Float64 "data: ${rad}"

j1_rad=" "
j2_rad=" "
j3_rad=" "
j4_rad=" "

if [ $joint -eq 1 ]
then
  j1_rad=$rad
elif [[ $joint -eq 2 ]]; then
  j2_rad=$rad
elif [[ $joint -eq 3 ]]; then
  j3_rad=$rad
else
  j4_rad=$rad
fi

echo "${j1_rad}, ${j2_rad},${j3_rad}, ${j4_rad}, Y" >> joints.txt
