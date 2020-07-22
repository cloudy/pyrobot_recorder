import os
import sys
import argparse
import numpy as np

#sys.path.append(os.path.dirname(__file__) + "/../ArmNavigation/n_joint_arm_to_point_control/")

import armnavigation.n_joint_arm_to_point_control.n_joint_arm_to_point_control as m

default_dir = 'armnavigation/n_joint_arm_to_point_control/trajectories/traj-0-test.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--file', help='File path', default=default_dir)
args = parser.parse_args()

data_f = args.file
data = np.loadtxt(data_f, delimiter=',')

m.show_animation = True
m.playback(data)

