# coding: utf-8
from environment import GraphicDisplay, Env
import numpy as np

'''Value iteration

Value iteration is `Off-policy` (IMHO). It means that the policy is fixed to
deterministic greedy policy (I dont know always this is right ...). We need
only the optimal value function V*(s) and then our greedy policy is optimal
too automatically.

This algorithm does not use policy when calculating the next value function V_{k+1}.
So we don't have to do policy improvement step which calculate next policy function.
'''

class ValueIteration(object):
    def __init__(self, env):
        self.env = env
        # self.value_table = [[0. for _ in range(self.env.width)] for _ in range(self.env.height)]
        self.value_table = np.zeros([self.env.height, self.env.width])
        self.discount_factor = 0.9

    def value_iteration(self):
        nvt = np.zeros([self.env.height, self.env.width])
        for state in self.env.get_all_states():
            if state == [2, 2]:
                nvt[2, 2] = 0.
                continue

            max_value = 0.
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                expected_value_for_action = reward + self.discount_factor*next_value
                if expected_value_for_action > max_value:
                    max_value = expected_value_for_action

            nvt[state[0], state[1]] = round(max_value, 2)

        self.value_table = nvt

    # 현재 가치 함수로부터 행동을 반환
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # 모든 행동에 대해 큐함수 (보상 + (감가율 * 다음 상태 가치함수))를 계산
        # 최대 큐 함수를 가진 행동(복수일 경우 여러 개)을 반환
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0], state[1]], 2)

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
