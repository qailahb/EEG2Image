# Script for verifying system build and dependencies 

import subprocess
import torch
import tensorflow as tf
import cv2
import numpy
import scipy
import matplotlib
import sklearn
import natsort

def get_system_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")

        for line in output.split("\n"):
            if "Cuda compilation tools" in line and "release" in line:
                return line.strip()
            
        return "CUDA version info not found in nvcc output."
    
    except FileNotFoundError:
        return "CUDA Toolkit not installed or not in PATH"
    
    except Exception as e:
        return f"Error running nvcc: {e}"

def main():

    # PyTorch
    print("PyTorch ----------------------------------------------")
    print(f"PyTorch version          : {torch.__version__}")
    print(f"PyTorch CUDA available   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version (torch)     : {torch.version.cuda}")
        print(f"cuDNN version (torch)    : {torch.backends.cudnn.version()}")
        print(f"GPU device               : {torch.cuda.get_device_name(0)}")

    # TensorFlow 
    print("TensorFlow ----------------------------------------------")
    print(f"TensorFlow version       : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TF GPU devices           : {gpus}")
    try:
        build_info = tf.sysconfig.get_build_info()
        print(f"CUDA version (TF build)  : {build_info.get('cuda_version', 'N/A')}")
        print(f"cuDNN version (TF build) : {build_info.get('cudnn_version', 'N/A')}")

    except Exception:
        print("TF CUDA/cuDNN info       : not available")

    # System CUDA Toolkit
    print("System CUDA Toolkit ----------------------------------------------")
    print(f"nvcc version             : {get_system_cuda_version()}")

    # Python Libraries
    print("Library Versions ----------------------------------------------")
    print(f"OpenCV version           : {cv2.__version__}")
    print(f"Numpy version            : {numpy.__version__}")
    print(f"SciPy version            : {scipy.__version__}")
    print(f"Matplotlib version       : {matplotlib.__version__}")
    print(f"Scikit-learn version     : {sklearn.__version__}")
    print(f"natsort version          : {natsort.__version__}")

if __name__ == "__main__":
    main()