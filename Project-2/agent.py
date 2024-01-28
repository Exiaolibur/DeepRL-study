from keras.layers import Dense
from keras.models import Sequential, load_model
import numpy as np


class Agent:
    def __init__(self, grid_size, epsilon = 1, epsilon_decay = 0.998, epsilon_end = 0.1, gamma = 0.99):
        self.grid_size = grid_size
        self.model = self.build_model()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma


    def build_model(self):
        model = Sequential(
            [Dense(128, activation='relu', input_shape=(self.grid_size ** 2,)),
            Dense(64, activation='relu'),
            Dense(4, activation='linear')]
        )


        model.compile(optimizer = "Adam", loss = "mse")

        return model


    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 4)

        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action


    def learn(self, experiences):
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        current_q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        target_q_values = current_q_values.copy()

        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]

            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        self.model.fit(states, target_q_values, epochs=1, verbose=0)

    def load(self, file_path):
        self.model = load_model(file_path)

    def save(self, file_path):
        self.model.save(file_path)