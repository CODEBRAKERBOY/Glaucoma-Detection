# TPU-Enabled Preprocessing Pipeline

import tensorflow as tf
from tensorflow.keras.models import Model
import warnings, logging, os

# Suppress warnings and logs
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TPU Setup
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU strategy initialized.")
except ValueError as e:
    print("TPU not available:", e)
    strategy = tf.distribute.get_strategy()  # Fallback to CPU/GPU

# Set Parameters and Random Seed
IMG_SIZE = (224, 224)
SEED = 999
tf.random.set_seed(SEED)
