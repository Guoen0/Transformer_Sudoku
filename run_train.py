from SudokuEnv import SudokuEnv
from Agent import Agent
from tools import save_config, load_config
import pickle

# ------------------------------------------------------------
is_continue_train = True       # 是否加载模型继续训练
suffix = "6x"                  # 选择 6x、9x 环境和模型，6x不需要验证小方格
# ------------------------------------------------------------

# 配置
if not is_continue_train:

    d_model = 512 if suffix == "9x" else 256 if suffix == "6x" else  64
    max_length = 200 if suffix == "9x" else 60 if suffix == "6x" else 10
    dff = d_model * 2

    num_iterations = 10000 if suffix == "9x" else 5000 if suffix == "6x" else 500

    config = {
        'env': {
            'size': int(suffix[0]),             # 6、9
        },
        'training': {
            'num_iterations': num_iterations,   # 总训练迭代次数
            'batch_size': 512,                  # batch size
            'learning_rate': 3e-4,              # SFT 学习率
            'num_show_play': 200,               # 观测间隔
            'num_print': 20,                    # 打印进度
        },
        'model': {
            'num_layers': 2,          # Transformer的层数
            'd_model': d_model,       # 每个词嵌入的维度    
            'num_heads': 2,           # 多头注意力机制的头数
            'dff': dff,               # 前馈神经网络的维度
            'dropout_rate': 0.1,      # Dropout率
        },
        'predict': {
            'max_length': max_length, # 推理时间步长度
            'temperature': 0.9,       # 探索温度
        }
    }
else:
    config = load_config(f"models/config_{suffix}.yaml")
    config['training'] = {
            'num_iterations': 2000,
            'batch_size': 512,
            'learning_rate': 3e-4,
            'num_show_play': 500,
            'num_print': 10,
        }

# 初始化
env = SudokuEnv(size=config['env']['size']) 
agent = Agent(env, config)

if env.size == 9:
    evaluation_data = pickle.load(open("data/evaluation_data_data_9x5k.pkl", "rb"))
    training_data = pickle.load(open("data/training_data_9x100k.pkl", "rb"))
elif env.size == 6:
    evaluation_data = pickle.load(open("data/evaluation_data_data_6x5k.pkl", "rb"))
    training_data = pickle.load(open("data/training_data_6x20k.pkl", "rb"))
elif env.size == 3:
    evaluation_data = pickle.load(open("data/evaluation_data_data_3x1k.pkl", "rb"))
    training_data = pickle.load(open("data/training_data_3x2k.pkl", "rb"))

agent.evaluation_data = evaluation_data
agent.training_data = training_data

if is_continue_train:
    agent.play()
    agent.policy_model.load_weights(f"models/policy_model_{suffix}.h5")

# 训练
agent.train()

# 保存
suffix = f"{config['env']['size']}x"
agent.policy_model.save_weights(f"models/policy_model_{suffix}.h5")
save_config(agent, f"models/config_{suffix}.yaml")