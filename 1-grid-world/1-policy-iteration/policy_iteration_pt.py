# coding: utf-8
import random
from environment import GraphicDisplay, Env


class PolicyIteration(object):
    def __init__(self, env):
        self.env = env
        self.value_table = [[0.] * env.width for _ in range(env.height)]
        # prior policy for up, left, down, right
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        # mark for final states
        self.policy_table[2][2] = []
        self.discount_factor = 0.9

    def policy_evaluation(self):
        pass

    def policy_improvement(self):

