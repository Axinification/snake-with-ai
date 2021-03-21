#Board Options
SIZE = 500 #Display size in pixels
TILES = 25 #Amount of tiles on the display

#Time controll
TIME_DELAY = 0 #50 #Change clock delay [ms]
CLOCK_TICK = 60 #10 #Change amount of fps

#Starting point
START_ROW = 15 #Change starting row
START_COLUMN = 10 #Change starting column

#Learning control
MAX_MEMORY = 100000 # Maximum amount of saved runs
BATCH_SIZE = 1000 # Maximum size of long training batch
LEARNING_RATE = 0.005 # The rate of bias change
EPSILON_DELTA = 100 # 0 randomness after x games


IS_INCREMENTING = True # On/Off option for gamma incrementing
GAMMA = 0.9 # Has to be less than 1. Lower discount rate strives for quick rewards and higher for the long term ones
GAMMA_LOW = 0.01 # Used for incrementing gamma over time // Set IS_INCREMENTING to True for it to work
#Reward system
REWARD = 10 # Both positive and negative
REWARD_MULTIPLIER = 5 # Multiplies positive reward
MINIMAL_REWARD = 10 # Minimal reward to give after time loss
TIME_PENALTY = 0.01 # This times frames after collecting apple or start will be substracted from apple reward

LOOP_TIME = 100 # Per snake segment