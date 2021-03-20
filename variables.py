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
LEARNING_RATE = 0.001 # The rate of bias change
EPSILON_DELTA = 80 # 0 randomness after 80 games
GAMMA = 0.9 # Has to be less than 1

#Reward system
REWARD = 10
REWARD_MULTIPLIER = 3