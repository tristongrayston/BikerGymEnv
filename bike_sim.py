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

TOTAL_ACTION_SPACE = 50 # Param for Training

# Math constants
A = 0.005
B = 222
CP = 4.2
MASS = 77 # in kilos
G = 9.81 # in netwons
R = 0.0046 # Rolling resistance constant
AIR = 0.00193 # Air resistance constant
DRAG_COEFF = 0.42 # Drag coefficient for a half circle



# Map
'''
Let's say we have a race that has two hills, one with 100 meter elevation, one with 150 meter elevation, and a net 0m elevation gain.

The first hill starts 2km in, plateaus at 7 km in, and dips down at 8 km in -> 10km in.

The second hill starts at 12km in, plateus at 17km in and dips back down at 19 km in.
'''
# Idea: 10 km race, 5 meter intervals, 2000 discrete states. 
x = np.rint(np.linspace(MIN_DISTANCE, MAX_DISTANCE, MAX_DISTANCE//INTERVAL_SIZE)).astype(int)

# Helper classes


class BikersEnv(gym.Env):
    '''
    The most simple and concrete way to get this up and running is to get it into OpenAI gym's format. 
    Because we're working with pygame, and because I expect this to train rather quickly, I'm going to go ahead and put the 
    render function inside the step function. This could very easily be removed and just thrown inside the game loop. 
    '''
    def __init__(self, width, height, render_mode):
        # game logic variables[45]

        self.metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}
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
        self.cur_AWC_pos = 0.5
        self.cur_vel = 0

        self.font = None

        # how many actions are made in a second. Definitely a tunable parameter.
        self.dt = 1

        ### gym.env variables ###

        # observation space
        self.observation_space = spaces.Dict(
            {
                "velocity": spaces.Box(BIKE_MIN, BIKE_MAX, shape=(1,), dtype=float),
                "total_power_capacity": spaces.Box(0, 1, shape=(1,), dtype=float),
                "total_distance_travelled": spaces.Box(MIN_DISTANCE, MAX_DISTANCE, shape=(1,), dtype=float)
            }
        )
        # action space - discretized
        # the action space takes values of intervals 1/20, with a max value of 1 and a min value of 0.
        self.action_space = spaces.Discrete(TOTAL_ACTION_SPACE)

        # current timestep and episode. 
        self.cur_ts = 0
        self.cur_episode = 0
        
        # Tracking variables
        self.cur_distance = MIN_DISTANCE

        
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
        '''
        PIPELINE FOR DEV:
        1. Make a tangible power output for an action
        2. Calculate the resistances on that power output
        3. Calculate how much AWC is recovered 
        4. Calculate how much distance was travelled. 
        
        '''
        # Get how much power we're outputting in watts given our agent's input.
        cur_p_o = self.power_output(action) 
        # Power output is measured in joules, so we effectively get kinetic energy. 

        # Calculate the resistance we face given that power output and our current state. 
        resistance = self.resistance(self.cur_distance, cur_p_o)

        # Calculate where we are on the AWC curve as a function of our current power output
        # Returns nothing because it modifies the class variable. 
        self.recovery_rate(cur_p_o) 

        # Calculate how far we travelled in the race as a function of our power and resistances. 
        self.calc_velocity(cur_p_o, resistance)
        print("distance travelled:", self.cur_vel)

        self.cur_distance += self.cur_vel.astype(int) // 5
        print(self.cur_distance)
        new_state = x[self.cur_distance]

        #print("###State###")
        #print("Power Output: ", cur_p_o)
        #print("angle: ", angle)
        #print("Distance Travelled: ", dist_travelled)
        #print("Current Distance Away: ", self.cur_distance)
        #print("new state: ", new_state)

        # current max distance = 1k or a value of 1000. This is due to change. 
        observation = {
                    "velocity" : self.cur_vel,
                    "total_power_capacity" : cur_p_o,
                    "total_distance_travelled" : self.cur_distance
                    }
        
        # existance penalty + how far away from total distance 
        if new_state == MAX_DISTANCE:
            reward = 10
        else:
            reward = -1 + new_state/MAX_DISTANCE

        terminated = 0

        # if distance <= 0
        if self.cur_distance == MAX_DISTANCE:
            reward = 100
            terminated = 1

        info = {}

        # render everything if we ask to render.
        if self.render_mode != "none":
            self.render(self.cur_vel)

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

    def resistance(self, state, power_output):
        '''
        
        '''
        angle = self.angles(state)
        #print("Angle!: ", angle)
        downward_force = G*MASS*np.cos(angle)

        rolling_resistance = R*G*MASS

        drag = 0.5*0.00193*(self.cur_vel**2)*DRAG_COEFF

        total_resistance = downward_force + rolling_resistance + drag

        return -total_resistance

    def power_output(self, action):
        '''
        Takes in the power coefficient which is the output from the agent. 

        We calculate our max power by determining our relative power, which is where our agent is on the curve.    
        Once we calculate our max power, we can get our power as a function of what the agent outputs. 

        '''
        max_AWC = self.output_power(self.cur_AWC_pos)*MASS
        print(action*(max_AWC + CP))
        return action*(max_AWC + CP)

    def calc_velocity(self, cur_KE, resistance):
        
        prev_KE = (1/2)*MASS*(self.cur_vel)**2
        total_force = (prev_KE + cur_KE)/2 - resistance
        if total_force > 0:
            self.cur_vel = np.sqrt(2*(total_force)/MASS)
        else:
            self.cur_vel = -np.sqrt(2*(-1*total_force)/MASS)

    def recovery_rate(self, current_power):
        
        # From Paper 1: Optimal pacing strategy modeling of cycling individual time trials
        if current_power//MASS > CP: # If current power is greater than CP, we want to be further on the curve.
            fatigue = ((current_power//MASS) - CP)
        else:
            fatigue = -(0.0879*(current_power/MASS) + 2.9214 - CP)

        print("RECOVERY RATE: ", fatigue)

        # The recovery rate works by tracking where we are on the curve. If we start to go towards zero, we need 
        # some place to stop. Since AWC' -> inf as x -> 0, I picked 0.1 as our max power.

        print("WHERE YOU ARE ON CURVE: ", self.cur_AWC_pos)
        if self.cur_AWC_pos + fatigue <= 0.5:
            self.cur_AWC_pos = 0.5
        else:
            self.cur_AWC_pos += fatigue
        
    def output_power(self, T):
        '''
        This power curve is derived from paper 1: Optimal pacing strategy modeling of cycling individual time trials.
        To get our W, or our maximum work capacity at any given time, we keep track of where we are on the curve. Because 
        we get the power output over one second, we take a right end reimann estimation to get our total area under the curve. 
        '''
        return CP + 170/(T+1)
    
    def angles(self, x):
        return np.piecewise(x, 
                [np.logical_and(0 <= x, x <= 1000), 
                np.logical_and(1000 < x, x <= 2000),
                np.logical_and(2000 < x, x <= 3000),
                np.logical_and(3000 < x, x <= 4000),
                np.logical_and(4000 < x, x <= 5000),
                np.logical_and(5000 < x, x <= 6000),
                np.logical_and(6000 < x, x <= 7000),
                np.logical_and(7000 < x, x <= 8000),
                np.logical_and(8000 < x, x <= 9000),
                np.logical_and(9000 < x, x <= 10000)],

                [lambda x : 1,
                    lambda x : -1,
                    lambda x : 1,
                    lambda x : -1,
                    lambda x : 1,
                    lambda x : -1,
                    lambda x : 1,
                    lambda x : -1,
                    lambda x : 1,
                    lambda x : -1]
                )

# Game logic functions / classes

class BG(pygame.sprite.Sprite):
    def __init__(self, groups):
        super().__init__(groups)
        ground_surface = pygame.image.load('images/background.png')
        scale_factor = 1.2
        #print(ground_surface.get_height())
        #print(ground_surface.get_width())

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
    velocity_i = []
    power_cap_i = []
    dist_i = []
    cur_dist = 0

    # for now our agent is random. 

    while cur_dist < 1980:
        
        action = np.random.randint(1, 50, 1)

        print(action)
        observation, reward, terminated, info = game.step(action)
        velocity_i.append(observation['velocity'])
        power_cap_i.append(observation['total_power_capacity'])
        cur_dist = observation['total_distance_travelled']
        dist_i.append(cur_dist)
        time.sleep(0.2)
        game.cur_ts += 1

    game.exit()
    plt.plot(velocity_i)
    plt.show()





