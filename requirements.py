import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib as plt
import sys
import shap
import optuna
import sklearn
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Python {sys.version}")
print(f"tensorflow=={tf.__version__}")
print(f"scikit-learn=={sklearn.__version__}")
print(f"numpy=={np.__version__}")
print(f"pandas=={pd.__version__}")
print(f"matplotlib=={plt.__version__}")
print(f"seaborn=={sns.__version__}")
print(f"shap=={shap.__version__}")
print(f"optuna=={optuna.__version__}")

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
