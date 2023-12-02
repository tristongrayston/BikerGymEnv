from courses import tenByOneKm
from math import exp
import gym
import pandas as pd
import numpy as np
from gym import spaces
import time
import json
from DQ_Agent import DQNAgent
import matplotlib.pyplot as plt
import torch

RIDER_AWC_MIN = 0.5 # in watts/kg
RIDER_AWC_MAX = 25 # in watts/kg

RIDER_VELOCITY_MIN = -28 # in m/s
RIDER_VELOCITY_MAX = 28 # in m/s

START_POSITION = 0 # in meters
COURSE_DISTANCE = 10000 # in meters

NUM_ACTIONS = 100

MAX_AWC = 9758 # in joules
CP_w = 234 # in watts
RIDER_MASS_kg = 70 # in kilos

G = 9.81 # in m/s^2
ROLLING_RESISTANCE_COEFF = 0.0046

EPISODES = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class BikersEnv(gym.Env):
    def __init__(self, course_fn):
        self.render_mode = None
        self.course_fn = course_fn

        self.observation_space = spaces.Dict(
            {
                "power_max_w": spaces.Box(low=0, high=3000, shape=(1,), dtype=float),
                "action": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "velocity": spaces.Box(low=RIDER_VELOCITY_MIN, high=RIDER_VELOCITY_MAX, shape=(1,), dtype=float),
                "position" : spaces.Box(low=START_POSITION, high=COURSE_DISTANCE, shape=(1,), dtype=float),
                "gradient": spaces.Box(low=-100, high=100, shape=(1,), dtype=float),
                "percent_complete" : spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "AWC": spaces.Box(low=RIDER_AWC_MIN, high=RIDER_AWC_MAX, shape=(1,), dtype=float),
                "rider_mass_kg": spaces.Box(low=0, high=100, shape=(1,), dtype=float),
            }
        )

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Environment meta variables
        # --------------------------
        self.cur_episode = 0

        # Agent variables
        # ---------------------------
        self.cur_AWC_j = MAX_AWC
        self.cur_velocity = 0
        self.cur_position = START_POSITION

    def step(self, action: int):

        # Normalize the action
        action = action/NUM_ACTIONS

        # Physiological calculations
        # -------------------------------------------------

        power_max_w = 7e-6*self.cur_AWC_j**2 + 0.0023*self.cur_AWC_j + CP_w

        power_agent_w = action*power_max_w

        if power_agent_w > CP_w:
            # Fatigue
            fatigue = -(power_agent_w - CP_w)
        else:
            # Recovery
            p_agent_adj_w = 0.0879*power_agent_w + 204.5
            fatigue = -(p_agent_adj_w - CP_w)

        self.cur_AWC_j = min(self.cur_AWC_j + fatigue, MAX_AWC)

        # Environment/Agent calculations
        # -------------------------------------------------

        # Resistance calculations
        # -----------------------

        slope_percent = self.course_fn(self.cur_position)
        slope_radians = np.arctan(slope_percent/100)

        horizontal_gravity_force = G*RIDER_MASS_kg*np.sin(slope_radians)

        rolling_resistance_force = G*RIDER_MASS_kg*ROLLING_RESISTANCE_COEFF

        drag_force = 0.19*(self.cur_velocity)**2

        total_resistance_w = (horizontal_gravity_force + rolling_resistance_force + drag_force)*self.cur_velocity

        # Velocity calculations
        # ---------------------

        cur_KE_j = (1/2)*RIDER_MASS_kg*(self.cur_velocity)**2

        final_KE_j = cur_KE_j - total_resistance_w + power_agent_w

        self.cur_velocity = np.sign(final_KE_j) * np.sqrt(2*abs(final_KE_j)/RIDER_MASS_kg)


        # Rider position calculations
        # ---------------------------

        new_position = self.cur_position + self.cur_velocity

        self.cur_position = min(new_position, COURSE_DISTANCE)

        # New Observation

        observation = {
                    "power_max_w": power_max_w,
                    "action": action,
                    "velocity" : self.cur_velocity,
                    "position" : self.cur_position,
                    "gradient": slope_percent,
                    "percent_complete" : self.cur_position/COURSE_DISTANCE,
                    "AWC": self.cur_AWC_j,
                    "rider_mass_kg": RIDER_MASS_kg,
                    }

        # Rewards and termination
        # -----------------------

        if self.cur_position >= COURSE_DISTANCE:
            reward = 1
        else:
            reward = -1 + (1/2)*(self.cur_velocity/RIDER_VELOCITY_MAX) + (1/2)*(power_max_w)/(7e-6*MAX_AWC**2 + 0.0023*MAX_AWC + CP_w)

        terminated = 0
        if self.cur_position >= COURSE_DISTANCE:
            reward = 1
            terminated = 1

        info = {
            **observation,
            "power" : power_agent_w,
            "reward": reward,
        }

        #print(json.dumps(info, indent=4, cls=NpEncoder))

        return observation, reward, terminated, info

    def render(self):
        return

    def reset(self):
        self.cur_episode += 1
        self.cur_position = START_POSITION
        self.cur_velocity = 0
        self.cur_AWC_j = MAX_AWC

        return {
            "power_max_w": 7e-6*self.cur_AWC_j**2 + 0.0023*self.cur_AWC_j + CP_w,
            "action": 0,
            "velocity" : 0,
            "position" : START_POSITION,
            "gradient": self.course_fn(self.cur_position),
            "percent_complete" : 0,
            "AWC": MAX_AWC,
            "rider_mass_kg": RIDER_MASS_kg,
        }

if __name__ == "__main__":
    game = BikersEnv(course_fn=tenByOneKm)
    agent = DQNAgent(6, NUM_ACTIONS)
    #agent.load('best_agent_1')
    data = []
    timesteps = []

    max_power_best = []
    rewards_best = []
    velocity_best = []
    lowest_timesteps = 10000000


    for e in range(EPISODES):
        max_power = []
        rewards = []
        velocity = []
        print('Epoch: ', e)
        timestep = 0
        observation = game.reset()
        obs_tensor = torch.Tensor(np.array([
                observation['power_max_w'],
                #observation['action'],
                observation['velocity'],
                observation['gradient'],
                observation['position'],
                observation['percent_complete'],
                observation['AWC'],
                #observation['rider_mass_kg']
        ])).to(device)
        terminated = 0

        while terminated == 0:
            # the agent takes tensor/vector values, so we change our dict
            timestep += 1
            prev_obs_tensor = obs_tensor

            action = agent.get_action(prev_obs_tensor)
            
            observation, reward, terminated, info = game.step(action)


            obs_tensor = torch.Tensor(np.array([
                    observation['power_max_w'],
                    #observation['action'],
                    observation['velocity'],
                    observation['gradient'],
                    observation['position'],
                    observation['percent_complete'],
                    observation['AWC'],
                    #observation['rider_mass_kg']
            ])).to(device)

            input_memory = (prev_obs_tensor, action, reward, obs_tensor)
            agent.replay_memory.store_memory(input_memory)

            max_power.append(obs_tensor[0])
            rewards.append(reward)
            velocity.append(obs_tensor[2])

            agent.learn()
            data.append(info)

        timesteps.append(timestep)

        if timestep < lowest_timesteps:
            lowest_timesteps = timestep
            max_power_best = max_power
            rewards_best = rewards
            velocity_best = velocity

        

    fig, ax = plt.subplots(4, 1, figsize=(14, 8))
    ax[3].plot(range(0, EPISODES, 1), timesteps)
    ax[2].plot(range(0, lowest_timesteps, 1), max_power_best)
    ax[1].plot(range(0, lowest_timesteps, 1), velocity_best)
    ax[0].plot(range(0, lowest_timesteps, 1), rewards_best)
    ax[3].set_ylabel("Timesteps ")
    ax[3].set_xlabel('Episode')
    ax[3].grid()
    ax[2].set_ylabel("W (P)")
    ax[2].set_xlabel("Timestep (s)")
    ax[2].grid()
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].grid()
    ax[0].set_ylabel("Rewards")
    ax[0].grid()

    plt.savefig(f'foo{e}.png', dpi=300)
    df = pd.DataFrame(data)
    df.to_csv(f'rider_data_{"random"}.csv', index=True)
    
    print(min(timesteps))

    plt.show()
    agent.save()

    