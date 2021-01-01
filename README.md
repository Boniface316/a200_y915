![GitHub repo size](https://img.shields.io/github/repo-size/Boniface316/a200_y915)
![GitHub top language](https://img.shields.io/github/languages/top/Boniface316/a200_y915)
![GitHub last commit](https://img.shields.io/github/last-commit/Boniface316/a200_y915)


# Computer vision and robotics

This objective of this project is to use CV to find the joint angles of the robot and identify target and then use concepts of robotics to follow the target.

# Prerequisite

* Ubuntu 18.04
* Python 2.7
* [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)

# Getting started

### Download and Installation
1.  Create a folder in your home directory, here we named the folder catkin ws, enter the folder and create another folder named src. You can do it via the terminal:
```
    $ mkdir catkin ws
    $ cd catkin ws
    $ mkdir src
```
2. Download or clone the repo to src folder using the instruction provided in [Github](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository).
3. Navigate back to catkin_ws in your terminal and enter ```catkin_make``` and ```source devel/setup.bash```
4. After that enter ```nano ~/.bashrc``` and add the following two lines:
```
    source /opt/ros/melodic/setup.bash
    source ∼/catkin ws/devel/setup.bash
```
5. Navigate to src folder within downloaded folder and convert the python scripts into executable files by entering the following code:
```
    chmod +x image1.py image2.py target move.py
```
6. Run the robot simulator using ```roslaunch [folder-name] spawn.launch```

![Robot simulator](https://github.com/Boniface316/a200_y915/blob/master/images/robot_orientation.png?raw=true)



## Authors
* **Moshen**
* **[Boniface Yogendran](https://github.com/Boniface316)**
* **[Sahar Atif](https://github.com/saharatif)**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
