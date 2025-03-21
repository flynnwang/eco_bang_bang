SINGLE_PLAER = False

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
MIN_TP_VAL = 30

MAX_GAME_STEPS = 505
MAX_MATCH_STEPS = 100

TEAM_POINTS_NORM = 100
TEAM_WIN_NORM = 5

# 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
# ACTION_NONE = 5  # TODO: when using sap, use a larger idx
ACTION_CENTER = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_LEFT = 4
MAX_MOVE_ACTION_IDX = ACTION_LEFT
ACTION_SAP = 5  # TODO: use other indices
MOVE_ACTION_NUM = 5
MOVE_ACTIONS_NO_CENTER = (ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT)

ACTION_ID_TO_NAME = {
    ACTION_CENTER: 'ACTION_CENTER',
    ACTION_UP: 'ACTION_UP',
    ACTION_RIGHT: 'ACTION_RIGHT',
    ACTION_DOWN: 'ACTION_DOWN',
    ACTION_LEFT: 'ACTION_LEFT',
    # ACTION_NONE: 'ACTION_NONE',
    ACTION_SAP: 'ACTION_SAP',
}

MIRRORED_ACTION = {
    ACTION_CENTER: ACTION_CENTER,
    ACTION_UP: ACTION_RIGHT,
    ACTION_RIGHT: ACTION_UP,
    ACTION_DOWN: ACTION_LEFT,
    ACTION_LEFT: ACTION_DOWN,
}

TRANSPOSED_ACTION = {
    ACTION_CENTER: ACTION_CENTER,
    ACTION_UP: ACTION_LEFT,
    ACTION_RIGHT: ACTION_DOWN,
    ACTION_DOWN: ACTION_RIGHT,
    ACTION_LEFT: ACTION_UP,
}

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

MIN_TEAM_WINS = 3

# N_BASELINE_EXTRA_DIM = 20
N_BASELINE_EXTRA_DIM = 14
N_TEAM_ACTIOR_EXTRA_DIM = 1

MAX_RELIC_NODE_NUM = 6
MAX_HIDDEN_RELICS_NUM = 37.5  # 25 * 0.25 * 6

MAX_VISION_REDUCTION = 3
MIN_UNIT_SENSOR_RANGE = 2
