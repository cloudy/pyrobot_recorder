import os
import sys
import numpy as np

#sys.path.append(os.path.dirname(__file__) + "/../ArmNavigation/n_joint_arm_to_point_control/")

import armnavigation.n_joint_arm_to_point_control.n_joint_arm_to_point_control as m

data_f = 'armnavigation/n_joint_arm_to_point_control/trajectories/traj-0-test.csv'
data = np.loadtxt(data_f, delimiter=',')

m.show_animation = True
m.playback(data)

