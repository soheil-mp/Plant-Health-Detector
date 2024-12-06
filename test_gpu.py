import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print GPU device information
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\nGPU Details:")
    for gpu in gpus:
        print(gpu)
    
    # Test GPU with a simple computation
    print("\nPerforming a test computation on GPU...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("\nMatrix multiplication result:", c.numpy())
else:
    print("\nNo GPU devices found.") 