import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from SudokuEnv import SudokuEnv
import pickle

# ------------------------------ 3x
env = SudokuEnv(size=3)
evaluate_data = env.roll_cold_start_data(1000)
with open("data/evaluation_data_3x1k.pkl", "wb") as f:
    pickle.dump(evaluate_data, f)
training_data = env.roll_cold_start_data(2000)
with open("data/training_data_3x2k.pkl", "wb") as f:
    pickle.dump(training_data, f)

# ------------------------------ 6x
env = SudokuEnv(size=6)
evaluate_data = env.roll_cold_start_data(5000)
with open("data/evaluation_data_6x5k.pkl", "wb") as f:
    pickle.dump(evaluate_data, f)
training_data = env.roll_cold_start_data(20000)
with open("data/training_data_6x20k.pkl", "wb") as f:
    pickle.dump(training_data, f)

# ------------------------------ 9x
env = SudokuEnv(size=9)
evaluate_data = env.roll_cold_start_data(5000)
with open("data/evaluation_data_9x5k.pkl", "wb") as f:
    pickle.dump(evaluate_data, f)
training_data = env.roll_cold_start_data(100000)
with open("data/training_data_9x100k.pkl", "wb") as f:
    pickle.dump(training_data, f)
