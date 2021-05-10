#Board Options
SIZE = 500 #Display size in pixels
TILES = 25 #Amount of tiles on the display
EASY_MODE = False #Walls on

#Time controll
TIME_DELAY = 0 #50 #Change clock delay [ms]
CLOCK_TICK = 120 #10 #Change amount of fps // Is starting 

#Starting point
START_ROW = 15 #Change starting row
START_COLUMN = 10 #Change starting column

#Learning control
MAX_MEMORY = 100000 # Maximum amount of saved runs
BATCH_SIZE = 1000 # Maximum size of long training batch
LEARNING_RATE = 0.001 # The rate of bias change
EPSILON_DELTA = 80 # 0 randomness after x games


IS_INCREMENTING = True # On/Off option for gamma incrementing
GAMMA = 0.9 # Has to be less than 1. Lower discount rate strives for quick rewards and higher for the long term ones
GAMMA_LOW = 0.1 # Used for incrementing gamma over time // Set IS_INCREMENTING to True for it to work
GAMMA_INCREMENT = 0.005 # Amount of gamma increment
#Reward system
REWARD = 10 # Both positive and negative // also minimal reward on time penalty
REWARD_MULTIPLIER = 3 # Multiplies positive reward
PENALTY_MULTIPLIER = 2 # Multiplies negative reward
TIME_PENALTY = 0.005 # This times frames after collecting apple or start will be substracted from apple reward
DIRECTION_REWARD = 0.5
STRAIGHT_LINE_REWARD = 0.6
COILING_PENALTY = -1

LOOP_TIME = 100 # Per snake segment

LOAD = False # Loading saved state

TRAIN_LOOPS = 1600 # Amount of games to play

LIVE_PLOT = False # Live plotting



