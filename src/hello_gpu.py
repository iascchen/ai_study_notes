import os

import tensorflow as tf

print('Tensorflow version : %s' % tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('GPU is ready : %s' % tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
# print('%s' % tf.config.list_physical_devices('GPU'))