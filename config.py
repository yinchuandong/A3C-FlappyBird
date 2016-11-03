GAME = 'flappy-bird'
STATE_DIM = 84
STATE_CHN = 4
ACTION_DIM = 2

LOCAL_T_MAX = 5  # repeat step size
RMSP_ALPHA = 0.99  # decay parameter for RMSProp
RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp
GAMMA = 0.99
ENTROPY_BETA = 0.0
MAX_TIME_STEP = 10 * 10**7

INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate
INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)

PARALLEL_SIZE = 1  # parallel thread size
USE_GPU = True
USE_LSTM = True
CHECKPOINT_DIR = 'checkpoints'

