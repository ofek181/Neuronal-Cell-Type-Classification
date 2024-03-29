import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_device() -> str:
    """
    prints out your physical devices.
    The user determines which device to use GPU/CPU by input.
    :return: device to be used.
    """
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

    cpus = tf.config.list_physical_devices('CPU')
    for cpu in cpus:
        print("Name:", cpu.name, "  Type:", cpu.device_type)

    device = input("Enter device (such as /device:GPU:0 or /device:CPU:0): ")
    return device
