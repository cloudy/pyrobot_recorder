"""
Class for controlling and plotting an arm with an arbitrary number of links.

Author: Daniel Ingram
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

class NLinkArm(object):
    def __init__(self, link_lengths, joint_angles, goal, show_animation, save_id=None, save_dir=None):
        self.show_animation = show_animation
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.plot_count = 0
        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]
        self.id = save_id if save_id is not None else time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = save_dir
        self.lim = sum(link_lengths)
        self.goal = np.array(goal).T
        print(save_id)
        print(save_dir)
        if save_dir is not None:
            try:
                os.mkdir(self.save_dir + self.id)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if show_animation:  # pragma: no cover
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('button_press_event', self.click)
            plt.ion()
            plt.show()
        
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T
        if self.show_animation:  # pragma: no cover
            self.plot()

    def plot(self):  # pragma: no cover
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        for i in range(self.n_links + 1):
            if i is not self.n_links:
                plt.plot([self.points[i][0], self.points[i + 1][0]],
                         [self.points[i][1], self.points[i + 1][1]], 'r-')
            plt.plot(self.points[i][0], self.points[i][1], 'ko')

        plt.plot(self.goal[0], self.goal[1], 'gx')

        plt.plot([self.end_effector[0], self.goal[0]], [
                 self.end_effector[1], self.goal[1]], 'g--')

        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        plt.draw()
        if self.save_dir is not None:
            plt.savefig(self.save_dir + self.id + '/plot-%05d.png' % self.plot_count)
        plt.pause(0.0001)
        self.plot_count += 1

    def click(self, event):
        self.goal = np.array([event.xdata, event.ydata]).T
        self.plot()
