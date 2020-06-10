"""
Inverse kinematics for an n-link arm using the Jacobian inverse method

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)
"""
import numpy as np
from random import random
from NLinkArm import NLinkArm

# Simulation parameters
Kp = 0.5
dt = 0.01
N_LINKS = 2
N_ITERATIONS = 10000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True
use_random_goal = False
goals = [[1.31, 1.1], [1.32, 1.1], [1.33, 1.05]] # examples if not using random goals
num_examples = 3


def main():  # pragma: no cover # doesn't get called. look at end
    """
    Creates an arm using the NLinkArm class and uses its inverse kinematics
    to move it to the desired position.
    """
    link_lengths = [1] * N_LINKS
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = [N_LINKS, 0]
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False
    while True:
        old_goal = np.array(goal_pos)
        goal_pos = np.array(arm.goal)
        end_effector = arm.end_effector
        errors, distance = distance_to_goal(end_effector, goal_pos)

        # State machine to allow changing of goal before current goal has been reached
        if state is WAIT_FOR_NEW_GOAL:
            if distance > 0.1 and not solution_found:
                joint_goal_angles, solution_found = inverse_kinematics(
                    link_lengths, joint_angles, goal_pos)
                if not solution_found:
                    print("Solution could not be found.")
                    state = WAIT_FOR_NEW_GOAL
                    arm.goal = end_effector
                elif solution_found:
                    state = MOVING_TO_GOAL
        elif state is MOVING_TO_GOAL:
            if distance > 0.1 and all(old_goal == goal_pos):
                joint_angles = joint_angles + Kp * \
                    ang_diff(joint_goal_angles, joint_angles) * dt
            else:
                state = WAIT_FOR_NEW_GOAL
                solution_found = False

        arm.update_joints(joint_angles)


def inverse_kinematics(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False


def get_random_goal():
    SAREA = 1.2 * N_LINKS
    return [SAREA * random() - SAREA / 2.0,
            SAREA * random() - SAREA / 2.0]


def animation():
    link_lengths = [1] * N_LINKS
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = get_random_goal() if use_random_goal else goals[0]
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False
    
    trajectory, trajectories = [joint_angles], []
    time, c_time, all_times = 0.0, [0.0], [] 
    i_goal = 0
    while True:
        old_goal = np.array(goal_pos)
        goal_pos = np.array(arm.goal)
        end_effector = arm.end_effector
        errors, distance = distance_to_goal(end_effector, goal_pos)

        # State machine to allow changing of goal before current goal has been reached
        if state is WAIT_FOR_NEW_GOAL:

            if distance > 0.1 and not solution_found:
                joint_goal_angles, solution_found = inverse_kinematics(
                    link_lengths, joint_angles, goal_pos)
                if not solution_found:
                    print("Solution could not be found.")
                    state = WAIT_FOR_NEW_GOAL
                    arm.goal = get_random_goal() if use_random_goal else goals[i_goal]
                    print(arm.goal)
                    print(get_random_goal())
                elif solution_found:
                    state = MOVING_TO_GOAL
        elif state is MOVING_TO_GOAL:
            if distance > 0.1 and all(old_goal == goal_pos):
                joint_angles = joint_angles + Kp * \
                    ang_diff(joint_goal_angles, joint_angles) * dt
                trajectory.append(joint_angles)
                time += dt
                c_time.append(time)
            else:
                joint_angles = np.array([0] * N_LINKS)
                state = WAIT_FOR_NEW_GOAL
                solution_found = False
                trajectories.append(np.array(trajectory))
                all_times.append(np.array(c_time))
                time = 0.0
                trajectory, c_time = [joint_angles], [time]
                i_goal += 1
                if i_goal >= num_examples:
                    break
                arm.goal = get_random_goal() if use_random_goal else goals[i_goal]
        
        arm.update_joints(joint_angles)
    for idx, traj in enumerate(trajectories):
        data = np.concatenate((all_times[idx].reshape(-1, 1), traj), axis=1)
        np.savetxt('trajectories/traj-%d.csv' % idx, data, delimiter=',')


def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    # main()
    animation()
