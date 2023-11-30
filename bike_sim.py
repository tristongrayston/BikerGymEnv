# Imports

from cgitb import text
from turtle import down
import numpy as np
import pygame
import matplotlib.pyplot as plt
import seaborn
from sys import exit
import gym
from gym import spaces
import time
import json
from DQ_Agent import DQNAgent
import torch

# CONSTANTS:

WIDTH = 800
HEIGHT = 500
clock = pygame.time.Clock()

# min/max biker velocity in m/s
BIKE_MIN = -1
BIKE_MAX = 1

# distance away from goal line
MIN_DISTANCE = 0
MAX_DISTANCE = 10000 # 1 km away, due to change.

INTERVAL_SIZE = 5

NUM_COEFFICIENTS = 50
EPISODES = 50

# Math constants
A = 0.005
B = 222
CP = 4.2
MASS = 77 # in kilos
G = 9.81 # in netwons
R = 0.0046 # Rolling resistance constant
AIR = 0.00193 # Air resistance constant
DRAG_COEFF = 0.42 # Drag coefficient for a half circle


# Possible Rider Positions
# ------------------------

x = np.rint(np.linspace(MIN_DISTANCE, MAX_DISTANCE, MAX_DISTANCE//INTERVAL_SIZE))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Environment Class
# -----------------

class BikersEnv(gym.Env):
    def __init__(self, width, height, render_mode):

        # Pygame variables
        self.metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = None

        self.width = width
        self.height = height
        self.screen = None

        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        self.background = None
        self.ground = None
        self.bike = None
        self.font = None

        self.observation_space = spaces.Dict(
            {
                "velocity": spaces.Box(low=BIKE_MIN, high=BIKE_MAX, shape=(1,), dtype=float),
                "total_power_capacity": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "total_distance_travelled": spaces.Box(low=MIN_DISTANCE, high=MAX_DISTANCE, shape=(1,), dtype=float)
            }
        )

        self.action_space = spaces.Discrete(NUM_COEFFICIENTS)

        # Environment meta variables
        # --------------------------
        self.dt = 1
        self.cur_timestep = 0
        self.cur_episode = 0

        # Agent variables
        # ---------------------------
        self.cur_AWC_pos = 0
        self.cur_velocity = 0
        self.cur_distance = MIN_DISTANCE
        self.terminated = 0


    def step(self, action: int):
        info = {}

        # regularize the action into something we can use (action -> [0, 50])
        action = action/NUM_COEFFICIENTS

        '''
        This power curve is derived from paper 1: Optimal pacing strategy modeling of cycling individual time trials.
        An important note, we shift our function over on the x axis to cap our maximum power output
        To get our W, or our maximum work capacity at any given time, we keep track of where we are on the curve. Because 
        we get the power output over one second, we integrate 
        '''

        max_AWC = (CP + 170/((self.cur_AWC_pos)+10))*MASS

        print("Max AWC: ", max_AWC)
        print("on curve: ", self.cur_AWC_pos)

        # How much power the agent is outputting in watts
        cur_agent_power_output = (max_AWC)*action

        # Calculate the resistance given power output and our current state.
        resistance = self.resistance(self.cur_distance)

        # Calculate where we are on the AWC curve as a function of our current power output
        # Returns nothing because it modifies the class variable.
        self.set_recovery_rate(cur_agent_power_output)

        # Calculate how far we travelled in the race as a function of our power and resistances.
        self.calc_velocity(cur_agent_power_output, resistance)

        self.cur_distance += self.cur_velocity.astype(int) // INTERVAL_SIZE
        if self.cur_distance >= MAX_DISTANCE//5:
            new_agent_position = x[(MAX_DISTANCE//5) - 1]
        else:
            new_agent_position = x[self.cur_distance]
        

        observation = {
                    "velocity" : self.cur_velocity,
                    "total_power_capacity" : cur_agent_power_output,
                    "total_distance_travelled" : self.cur_distance
                    }

        # existance penalty + how far away from total distance
        if new_agent_position >= MAX_DISTANCE:
            reward = 10
        else:
            reward = -1 + new_agent_position/MAX_DISTANCE

        self.terminated = 0

        # if distance <= 0
        if self.cur_distance >= MAX_DISTANCE//5:
            self.terminated = 1

        # render everything if we ask to render.
        if self.render_mode != None:
            self.render(self.cur_velocity)

        self.cur_timestep += 1

        print(json.dumps({
            **observation,
            "action": action,
            "reward": reward,
        }, indent=4, cls=NpEncoder))

        return observation, reward, self.terminated, info


    def render(self, vel):
        if self.screen == None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Comic Sans MS', 20)
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.background = BG(self.all_sprites)
            self.ground = Ground(self.all_sprites, 6)
            self.bike = Bike(self.all_sprites, 0.2)

        self.all_sprites.update(self.dt, vel)
        self.all_sprites.draw(self.screen)

        text_surface_ts = self.font.render(f'Episode: {self.cur_episode}', False, (0, 0, 0))
        text_surface_ep = self.font.render(f'Timestep: {self.cur_timestep}', False, (0, 0, 0))
        self.screen.blit(text_surface_ts, (0,0))
        self.screen.blit(text_surface_ep, (0,30))
        pygame.display.update()

    def reset(self):
        self.dt = 1
        self.cur_timestep = 0
        self.cur_episode = 0

        # Agent variables
        # ---------------------------
        self.cur_AWC_pos = 0
        self.cur_velocity = 0
        self.cur_distance = MIN_DISTANCE
        self.terminated = 0
        return ({ "velocity" : self.cur_velocity,
                    "total_power_capacity" : 0,
                    "total_distance_travelled" : self.cur_distance
                    }, self.terminated)

    def resistance(self, state):
        angle = self.angles(state*INTERVAL_SIZE)

        downward_force = G*MASS*np.sin((angle)*(np.pi/180))

        rolling_resistance = R*G*MASS

        drag = 0.0193*(self.cur_velocity**2)

        total_resistance = downward_force + rolling_resistance + drag

        return -total_resistance * self.cur_velocity


    def calc_velocity(self, cur_KE, resistance):

        prev_KE = (1/2)*MASS*(self.cur_velocity)**2
        total_force = 0.9*prev_KE + cur_KE + resistance
        if total_force > 0:
            self.cur_velocity = np.sqrt(2*(total_force)/MASS)
        else:
            self.cur_velocity = -np.sqrt(2*(-1*total_force)/MASS)

    def set_recovery_rate(self, cur_power):
        # If current power is greater than CP
        # we want to be further on the curve.
        if cur_power/MASS > CP:
            fatigue = ((cur_power/MASS) - CP)
        else:
            fatigue = -(0.0879*(cur_power/MASS) + 2.9214 - CP)

        print(fatigue)
        if self.cur_AWC_pos + fatigue <= 0:
            self.cur_AWC_pos = 0
        else:
            self.cur_AWC_pos += fatigue

    def angles(self, x):
        return np.piecewise(x,
                [np.logical_and(0 <= x,  x <= 1000),
                np.logical_and(1000 < x, x <= 2000),
                np.logical_and(2000 < x, x <= 3000),
                np.logical_and(3000 < x, x <= 4000),
                np.logical_and(4000 < x, x <= 5000),
                np.logical_and(5000 < x, x <= 6000),
                np.logical_and(6000 < x, x <= 7000),
                np.logical_and(7000 < x, x <= 8000),
                np.logical_and(8000 < x, x <= 9000),
                np.logical_and(9000 < x, x <= 10000)],

                [lambda x : 2.8624,
                    lambda x : -2.8624,
                    lambda x : 2.8624,
                    lambda x : -2.8624,
                    lambda x : 2.8624,
                    lambda x : -2.8624,
                    lambda x : 2.8624,
                    lambda x : -2.8624,
                    lambda x : 2.8624,
                    lambda x : -2.8624]
                )

# Game logic functions / classes

class BG(pygame.sprite.Sprite):
    def __init__(self, groups):
        super().__init__(groups)
        ground_surface = pygame.image.load('images/background.png')
        scale_factor = 1.2

        full_height = ground_surface.get_height() * scale_factor
        full_width = ground_surface.get_width() * scale_factor
        full_image = pygame.transform.scale(ground_surface, (full_width, full_height))

        self.image = pygame.Surface((full_width*2, full_height))
        self.image.blit(full_image, (0, 0))
        self.image.blit(full_image, (full_width, 0))

        self.rect = self.image.get_rect(topleft = (0, 0))
        self.pos = pygame.math.Vector2(self.rect.topleft)

    def update(self, dt, vel):

        #self.rect.x = round(self.pos.x)
        if self.rect.centerx <= 0:
            self.pos.x = 0

        self.pos.x -= vel * dt
        self.rect.x = round(self.pos.x)

class Ground(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups)
        self.scale_factor = scale_factor
        # image


        ground_surface = pygame.image.load('images/backgrounds.png').convert_alpha()

        # for line
        full_height = ground_surface.get_height()
        full_width = ground_surface.get_width() * scale_factor
        full_image = pygame.transform.scale(ground_surface, (full_width, full_height))

        self.image = pygame.transform.scale(full_image, pygame.math.Vector2(full_image.get_size()))

        # position
        self.rect = self.image.get_rect(bottomleft = (0,HEIGHT))
        self.pos = pygame.math.Vector2(self.rect.topleft)

    def update(self, dt, vel):
        if self.rect.centerx <= 0:
            self.pos.x = 0

        self.pos.x -= vel * dt
        self.rect.x = round(self.pos.x)

class Bike(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups)

        # image
        self.import_frames(scale_factor)
        self.frame_index = 0
        self.image = self.frames[self.frame_index]

        # rect
        self.rect = self.image.get_rect(midleft = (WIDTH / 20, 400))
        self.pos = pygame.math.Vector2(self.rect.topleft)

        # movement
        self.gravity = 20 # <-- change this
        self.direction = 0

    def import_frames(self, scale_factor):
        self.frames = []
        for i in range(27):
            if i < 10:
                surf = pygame.image.load(f'images/biker_frames/frame_0{i}_delay-0.04s.gif')
            else:
                surf = pygame.image.load(f'images/biker_frames/frame_{i}_delay-0.04s.gif')

            scaled_surface = pygame.transform.scale(surf, pygame.math.Vector2(surf.get_size()) * scale_factor)
            self.frames.append(scaled_surface)

    def animate(self,dt):
        self.frame_index += 10 * dt
        if self.frame_index >= len(self.frames):
            self.frame_index = 0
        self.image = self.frames[int(self.frame_index)]

    def update(self, dt, vel):
        self.animate(dt/50)




# Game loop

if __name__ == "__main__":
    game = BikersEnv(HEIGHT, WIDTH, "human")
    agent = DQNAgent(3, NUM_COEFFICIENTS)
    rewards = []
    time_per_ep = []
    cur_dist = 0
    terminated = 0

    for e in range(20):
        print("TIMESTEP: ", e)
        observation, terminated = game.reset()
        observation = [
                observation['velocity'],
                observation['total_power_capacity'],
                observation['total_distance_travelled']
        ]
        ep_reward = 0
        
        while terminated == 0:
            
            prev_observation = observation

            action = agent.get_action(prev_observation)
            observation, reward, terminated, info = game.step(action)
            cur_dist = observation['total_distance_travelled']

            observation = [
                observation['velocity'],
                observation['total_power_capacity'],
                observation['total_distance_travelled']
            ]

            input_memory = (prev_observation, action, reward, observation)
            agent.replay_memory.store_memory(input_memory)
            ep_reward += reward

            agent.learn()

        rewards.append(ep_reward)
        time_per_ep.append(game.cur_timestep)
        print("FINISHED TIMESTEP: ", e)

    
    #power_cap_i = np.array(power_cap_i)/MASS
    #plt.plot(power_cap_i, label='power cap at time t')
    #plt.plot(velocity_i, label='velocity at time t')
    plt.plot(rewards, label='rewards')
    plt.plot(time_per_ep, label='time_per_ep')
    plt.legend()
    plt.grid()
    plt.show()