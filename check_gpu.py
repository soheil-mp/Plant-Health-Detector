import tensorflow as tf
import sys

print("Python version:", sys.version)
print("\nTensorFlow version:", tf.__version__)

# Check for GPU availability
print("\nGPU Information:")
physical_devices = tf.config.list_physical_devices()
print("\nAll available devices:")
for device in physical_devices:
    print(f"- {device.device_type}: {device.name}")

print("\nGPU devices:")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        print(f"- {device.name}")
    print(f"\nTotal GPU(s) available: {len(gpu_devices)}")
else:
    print("No GPU devices found.")

# Try to get more detailed GPU info if available
try:
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    print("\nDetailed device information:")
    for device in local_devices:
        print(f"\nDevice: {device.name}")
        print(f"Device type: {device.device_type}")
        if device.device_type == 'GPU':
            print(f"Memory limit: {device.memory_limit / (1024**3):.2f} GB")
except Exception as e:
    print(f"\nCould not get detailed GPU information: {e}") 