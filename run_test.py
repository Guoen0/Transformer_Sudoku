from tools import load_config
from Agent import Agent
from SudokuEnv import SudokuEnv

# ------------------------------------------------------------
continue_suffix = "9x"       # 选择 6x、9x 环境和模型，6x不需要验证小方格
# ------------------------------------------------------------

# 加载
config = load_config(f"./models/config_{continue_suffix}.yaml")

env = SudokuEnv(size = config['env']['size'])
agent = Agent(
    env = env,
    config = config
)

suffix = f"{config['env']['size']}x"
agent.policy_model.load_weights(f"models/policy_model_{suffix}.h5")
agent.policy_model.summary()

# 测试
ret = agent.play(batch_size=512, run_type="test")
right_ret = [d for d in ret if d['reward'] == 1]
success = len(right_ret)/len(ret)

# 观察正确答案中最长的几个
max_lengths_answer_data = sorted(right_ret, key=lambda d: len(d['answer']), reverse=True)[:2]
for data in max_lengths_answer_data:
    question = data['question']
    answer = data['answer']

    actions = []
    for token in answer:
        action = env._token_to_action(token)
        actions.append(action)
    print("预测的动作：", actions)

    env.play(state_tokens=question, action_tokens=answer, is_print=True)

print(f"成功率: {success}")

