# Imports

from cgitb import text
import numpy as np
import pygame
import matplotlib.pyplot as plt
import seaborn 
from sys import exit
import gym
from gym import spaces

# CONSTANTS:

WIDTH = 800
HEIGHT = 500
clock = pygame.time.Clock()


# min/max biker velocity in m/s
BIKE_MIN = -1
BIKE_MAX = 1

# distance away from goal line
MIN_DISTANCE = 0
MAX_DISTANCE = 1000 # 1 km away, due to change. 

TOTAL_ACTION_SPACE = 50 # 

# Helper classes


class BikersEnv(gym.Env):
    '''
    The most simple and concrete way to get this up and running is to get it into OpenAI gym's format. 
    Because we're working with pygame, and because I expect this to train rather quickly, I'm going to go ahead and put the 
    render function inside the step function. This could very easily be removed and just thrown inside the game loop. 
    '''
    def __init__(self, width, height, render_mode):
        # game logic variables[45]

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.width = width
        self.height = height
        self.screen = None

        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        self.background = None
        self.ground = None
        self.bike = None
        self.cur_total_power = 100

        self.font = None

        # how many actions are made in a second. Definitely a tunable parameter.
        self.dt = 1/5

        ### gym.env variables ###

        # observation space
        self.observation_space = spaces.Dict(
            {
                "velocity": spaces.Box(BIKE_MIN, BIKE_MAX, shape=(1,), dtype=float),
                "power_output": spaces.Box(0, 1, shape=(1,), dtype=float),
                "fatigue": spaces.Box(0, 1, shape=(1,), dtype=float),
                "resistance": spaces.Box(-1000, 1000, shape=(1,), dtype=float),
                "distance": spaces.Box(MIN_DISTANCE, MAX_DISTANCE, shape=(1,), dtype=float)
            }
        )
        # action space - discretized
        # the action space takes values of intervals 1/20, with a max value of 1 and a min value of 0.
        self.action_space = spaces.Discrete(TOTAL_ACTION_SPACE)

        # current timestep and episode. 
        self.cur_ts = 0
        self.cur_episode = 0
        
        # Tracking variables
        self.cur_distance = MAX_DISTANCE

        
    def step(self, action: int):
        '''
        Actions are discrete values from 0/20 for convenience. 
        Step function contains the main logic of the environment. Meaning,
        it calculates what state you're in after given an environment. 
        
        First and formost, we need to know current velocity, so we use all our helper functions to find
        that out.

        Then, we calculate reward.

        Then we render the environment (normally this would be in a different step, but this is easier.)

        Then we pass back everything. 
        '''
        # regularize the action into something we can use (action -> [0, 50])
        action = action/TOTAL_ACTION_SPACE


        # Calculate velocity given out power_coefficient.
        cur_resistance = resistance()
        cur_fatigue = fatigue() # todo
        rr = recovery() # todo
        cur_p_o = power_output() # todo
        cur_vel = velocity(action) # todo

        self.cur_distance -= cur_vel

        # current max distance = 1k or a value of 1000. This is due to change. 
        observation = {
                    "velocity" : cur_vel,
                    "power_output" : cur_p_o,
                    "fatigue" : cur_fatigue,
                    "resistance" : cur_resistance,
                    "distance" : self.cur_distance
                    }
        
        # existance penalty
        reward = -1
        terminated = 0

        # if distance <= 0
        if self.cur_distance == 0:
            reward = 100
            terminated = 1

        info = {}

        # render everything
        self.render(cur_vel)

        return observation, reward, terminated, info
    

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
        text_surface_ep = self.font.render(f'Timestep: {self.cur_ts}', False, (0, 0, 0))
        self.screen.blit(text_surface_ts, (0,0))
        self.screen.blit(text_surface_ep, (0,30))
        pygame.display.update()

        
    def reset(self):
        self.distance = MAX_DISTANCE

#### Helper functions/classes ####

# Math modelling functions/classes

def resistance():
    '''
    Takes in some output from power_output. 
    '''
    return 0

def power_output():
    '''
    Takes in the power coefficient which is the output from the agent.      
    '''
    return 100

def force():
    ''' '''
    return 0

def velocity(power_coefficient: float) -> float:
    '''
    This function is supposed to calculate forward velocity. Which means it considers
    fatigue, recovery, and resistance values -> (gravity, wind resistance, etc) to produce a
    float value for how fast it is going forward. This is on the agenda of things to sort out. 
    '''
    
    return BIKE_MAX*power_coefficient

def fatigue():
    return 0

def recovery():
    return 0

# Game logic functions / classes

class BG(pygame.sprite.Sprite):
    def __init__(self, groups):
        super().__init__(groups)
        ground_surface = pygame.image.load('images/background.png')
        scale_factor = 1.2
        print(ground_surface.get_height())
        print(ground_surface.get_width())

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
    game.reset()
    rewards = []

    # for now our agent is random. 

    while True:
        
        action = np.random.randint(0, 50, 1)

        print(action)
        observation, reward, terminated, info = game.step(action)
        rewards.append(reward)
        game.cur_ts += 1

