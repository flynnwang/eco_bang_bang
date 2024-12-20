PLAYER0 = 'player_0'
PLAYER1 = 'player_1'

UNITS_ACTION = 'units_action'

MAX_UNIT_NUM = 16

MAP_WIDTH = 24
MAP_HEIGHT = 24
MAP_SIZE = MAP_WIDTH

CELL_UNKONWN = 0
CELL_SPACE = 1
CELL_NEBULA = 2
CELL_ASTERIOD = 3
N_CELL_TYPES = CELL_ASTERIOD + 1

RELIC_NB_SIZE = 5

TEAM_POINT_MASS = 100
NON_TEAM_POINT_MASS = -100

MAX_GAME_STEPS = 505
MAX_MATCH_STEPS = 100

# 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
ACTION_CENTER = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_LEFT = 4
ACTION_SAP = 5
MOVE_ACTION_NUM = 5

DIRECTIONS = [
    [0, 0],  # Do nothing
    [0, -1],  # Move up
    [1, 0],  # Move right
    [0, 1],  # Move down
    [-1, 0],  # Move left
]

MAX_MOVE_COST = 5
MAX_SENSOR_RANGE = 4

MAX_ENERTY_PER_TILE = 20
MAX_UNIT_ENERGY = 400
