import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import Adam

from keras.models import load_model

from collections import deque


class DQNAgent:
    def __init__(self, env, player, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.01, tau=0.05, model_path="", target_model_path=""):
        self.env = env
        self.player = player
        self.win_count = 0

        self.memory = deque(maxlen=2000)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        if model_path == "":
            self.model = self.create_model()
        else:
            self.model = load_model(model_path)

        if target_model_path == "":
            self.target_model = self.create_model()
        else:
            self.target_model = load_model(target_model_path)
         
    def create_model(self):
        model = Sequential()
        # input_shape = self.env.observation_space(self.player).shape
        # model.add(LSTM(64,input_shape=(1, 16)))
        input_dim = len(self.env.observation_space(self.player))
        model.add(Dense(128, activation="relu", input_dim=input_dim))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="softmax"))
        
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model
    
    def act(self, observation):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.valid_actions)
        predictions = self.model.predict(np.array([observation]))[0]
        for idx, prediction in enumerate(predictions):
            if idx not in self.env.valid_actions:
                predictions[idx] = 0
        
        #nomalizing
        sum_predictions = sum(predictions)
        if sum_predictions != 0:
            prescaler = 1/sum(predictions)
            predictions = [prescaler * prediction for prediction in predictions]
            action = np.argmax(predictions)
        else:
            action = 0
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            observation, action, reward, new_observation, done = sample
            # observation = observation.reshape(1, 1, 16)
            observation = np.array([observation])
            target = self.target_model.predict(observation)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(observation)[0])
                target[0][action] = reward + Q_future * self.gamma
        self.model.fit(observation, target, epochs=1, verbose=0)
        # print(self.model.summary())
        # exit()
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def save_target_model(self, fn):
        self.target_model.save(fn)
