GAME = 'flappy-bird'
STATE_DIM = 84
STATE_CHN = 4
ACTION_DIM = 2

LOCAL_T_MAX = 5  # repeat step size
RMSP_ALPHA = 0.99  # decay parameter for RMSProp
RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp
GAMMA = 0.99
ENTROPY_BETA = 0.000001  # 0.01 for FFNet
MAX_TIME_STEP = 10 * 10**7

INITIAL_ALPHA_LOW = 1e-6   # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-4   # log_uniform high limit for learning rate
INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)

PARALLEL_SIZE = 1  # parallel thread size, please start game_server first
USE_GPU = True
USE_LSTM = True
LSTM_UNITS = 256

CHECKPOINT_DIR = 'tmp/checkpoints'
LOG_FILE = 'tmp/a3c_log'

