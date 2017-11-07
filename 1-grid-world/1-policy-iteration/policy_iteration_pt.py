# coding: utf-8
import random
from environment import GraphicDisplay, Env

'''Policy iteration

policy iteration start from random policy. The policy is used for evaluating value function,
and the value function is also used for improve current policy.
'''


class PolicyIteration(object):
    def __init__(self, env):
        self.env = env
        self.value_table = [[0. for _ in range(env.width)] for _ in range(env.height)]
        # prior policy for up, left, down, right
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25] for _ in range(env.width)] for _ in range(env.height)]
        # mark for final states
        self.policy_table[2][2] = []
        self.discount_factor = 0.9

    def policy_evaluation(self):
        '''Evaluate policy: calculate V_pi(s)
        pi is current policy. '''

        nvt = [[0.] * self.env.width for _ in range(self.env.height)] # next value table
        for state in self.env.get_all_states():
            if state == [2, 2]: # final state
                nvt[2][2] = 0. # [?] final state value = 0?
                continue

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                # [?] If transition_prob is not 1?
                # transition_prob = self.env.get_transition_prob(state, action)
                policy = self.get_policy(state)[action]
                expected_value_for_action = reward + self.discount_factor*next_value
                nvt[state[0]][state[1]] += policy * expected_value_for_action

            nvt[state[0]][state[1]] = round(nvt[state[0]][state[1]], 2)

        self.value_table = nvt

    def policy_improvement(self):
        npt = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                npt[2][2] = []
                continue

            Q_list = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                next_value = self.get_value(next_state)
                reward = self.env.get_reward(state, action)
                expected_value_for_action = reward + self.discount_factor*next_value # Q(s,a)
                Q_list.append(expected_value_for_action)

            m = max(Q_list)
            mc = sum([q == m for q in Q_list])
            prob = 1. / mc

            for i in range(4):
                if Q_list[i] == m:
                    npt[state[0]][state[1]][i] = prob
                else:
                    npt[state[0]][state[1]][i] = 0.

        self.policy_table = npt

    # this function is required for env (does not used in this policy agent)
    # 특정 상태에서 정책에 따른 행동을 반환
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    def get_policy(self, state):
        if state == [2, 2]:
            return 0.
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2) # calc only to second decimal


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
