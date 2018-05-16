import pandas as pd
import numpy as np

class RL:
    def __init__(self, Actions, Epsilon = .9, Lambda = .9, Alpha = .1):
        self.Epsilon = Epsilon
        self.Lambda  = Lambda
        self.Alpha   = Alpha
        self.Actions = Actions
        self.q_table = pd.DataFrame(columns=Actions)

    def actor(self, state):
        self.check_state(state)
        q_value = self.q_table.loc[[state]].iloc[0]
        if np.random.uniform(0, 1) > self.Epsilon or q_value.all() == 0:
            action = np.random.choice(self.Actions)
        else:
            q_value = q_value.reindex(np.random.permutation(self.Actions))
            action = q_value.argmax()

        return action

    def learn(self, state, action, reward, next_state, next_action):
        self.q_table.ix[state, action] += self.Alpha * \
                                          (reward + self.Lambda * self.q_table.ix[next_state, next_action] - \
                                          self.q_table.ix[state, action])

    def check_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0.] * len(self.Actions),
                    index=self.Actions,
                    name=state
                )
            )
