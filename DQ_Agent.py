import numpy as np
import random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# HYPERPARAMS
REPLAY_BUFFER_MAX_LENGTH = 100000
BUFFER_BATCH_SIZE = 72
BATCH_SIZE = 64
GAMMA = 1
TARGET_UPDATE = 32
EPSILON_START = 0.9 # Epsilon Start
EPSILON_END = 0.05 # Epsilon End
EPSILON_DECAY = 0.00001 # Epsilon Decay
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_from_path = "best_agent"
save_to_path = "best_agent"


class ReplayBuffer():

    def __init__(self):
        self.memory = deque(maxlen = REPLAY_BUFFER_MAX_LENGTH)

    def store_memory(self, experience: tuple) -> None:
        self.memory.append(experience)

    def collect_memory(self) -> tuple:
        return self.memory.pop()
        # if np.random.rand() < 0.8:
        #     return self.memory.pop()
        # else:
        #     return rnd.choice(self.memory)

    def erase_memory(self):
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent():
    def __init__(self, input_dims, output_dims):
        self.output_dims = output_dims
        self.input_dims = input_dims
        #self.observation_space = observation_space

        # actor and target models:
        self.model = self.create_model(self.input_dims, self.output_dims)
        self.target_model = self.create_model(self.input_dims, self.output_dims)
        self.replay_memory = ReplayBuffer()
        self.update_target_counter = 0
        self.total_actions_taken = 0
        self.actions_taken = 0


        self.optimizer = torch.optim.Adam(self.model.parameters(), LR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, input_dims, output_dims):
        # Very simple model. Not sure that we need anything more complex than this.
        model = nn.Sequential(
            nn.Linear(input_dims, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dims),
        )
        return model

    # Method for predicting an action
    def get_action(self, state):
        #print(state.shape)
        #state = np.transpose(state, (2, 0, 1))
        #print(state.shape)
        self.actions_taken += 1
        self.total_actions_taken += 1
        EPSILON = (EPSILON_START)*np.exp(-EPSILON_DECAY*self.total_actions_taken) # EPSILON_END

        with torch.no_grad():
            #input = state_array.reshape(1)
            input_state = torch.Tensor(state).to(device)
            #input_state = input_state.permute(2, 0, 1)
            q_values = self.target_model.forward(input_state)

        #print(q_values)
        if rnd.random() < EPSILON + EPSILON_END:
        #if rnd.random() < 0.25:
            # Pick a random action
            action = torch.tensor([rnd.randrange(0, self.output_dims)])
        else:
            action = torch.argmax(q_values)
        #print(action)


        return action.item()

    def learn(self):
        ''' We pretty much strictly learn from the memory, not
            specifically from the current experience. '''

        # We just pass through the learn function if the batch size has not been reached.
        if self.replay_memory.__len__() < BUFFER_BATCH_SIZE:
            return

        # DQN uses MSE and stochastic gradient descent
        criterion = nn.SmoothL1Loss()

        state = []
        action = []
        reward = []
        next_state = []
        for _ in range(BATCH_SIZE):
            s, a, r, n = self.replay_memory.collect_memory()

            state.append(s)
            action.append(a)
            reward.append(torch.tensor(r, dtype=torch.float32).to(device))
            next_state.append(n)

        # Convert list of tensors to tensor.

        state_tensors = torch.stack(state)
        new_state_tensors = torch.stack(next_state)

        rewards = torch.stack(reward)

        # One hot encoding our actions.
        action = torch.eye(self.output_dims)[action].to(device)

        #Find our predictions
        with torch.no_grad():
            # This gets the maximum possible Q value of our next turn
            target_predictions = self.target_model(new_state_tensors).max(dim=1)[0]

        # this gets the models assessed Q value of the current turn.
        model_predictions = self.model(state_tensors)*action
        model_predictions = torch.sum(model_predictions, dim=1)

        #model_predictions = self.model(state_tensors).max(dim=1)[0][action] # .max(dim=1)[action]

        # Calculate our target
        target = rewards + GAMMA*target_predictions

        # Calculate MSE Loss
        loss = criterion(model_predictions, target)

        #print(loss)

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip Gradients

        # By value
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

        # By norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)


        self.optimizer.step()

        #print("learning..")
        self.update_target_counter += 1
        #print(self.update_target_counter)
        if self.update_target_counter % TARGET_UPDATE == 0:
            print('updating...')
            self.target_model.load_state_dict(self.model.state_dict())
            print('Current Epsilon, ', ((EPSILON_START)*np.exp(-EPSILON_DECAY*self.total_actions_taken)))

    def save(self):
        torch.save(self.target_model.state_dict(), save_to_path)

    def load(self, path):
        self.target_model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path))
        

        print(f'CURRENT MODEL VERSION: {path}')
        pass

