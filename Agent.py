import random
import tensorflow as tf
from collections import deque
from SudokuEnv import SudokuEnv
from Transformer import Transformer
from tools import truncate_end, get_padded_batch_data

class Agent:
    def __init__(self, env: SudokuEnv, config: dict):
        # 初始化标识和维度
        self.env = env
        self.vocab = env.vocab
        self.input_vocab_size = len(self.vocab)     # 词汇表大小
        self.target_vocab_size = len(self.vocab) 
        self.START_ID = tf.constant(self.vocab.index("<START>"), dtype=tf.int64)
        self.END_ID = tf.constant(self.vocab.index("<END>"), dtype=tf.int64)
        self.PAD_ID = tf.constant(self.vocab.index("<PAD>"), dtype=tf.int64)
        self.config = config

        self.evaluation_data = []
        self.training_data = []
        self.memory = []

        self.policy_model = self._build_policy_model(config)
        self._optimizer_compute_learning_rate()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.current_temperature = config["predict"]["temperature"]
        
    def _build_policy_model(self, config):
        model = Transformer(
            num_layers=config['model']['num_layers'],
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            dff=config['model']['dff'],
            input_vocab_size=self.input_vocab_size,
            target_vocab_size=self.target_vocab_size,
            maximum_position_encoding=config['predict']['max_length'] + 1,
            rate=config['model']['dropout_rate']
        )
        # 构建模型
        dummy_input = tf.zeros((1, 1), dtype=tf.int64)
        model((dummy_input, dummy_input))
        return model

    def _optimizer_compute_learning_rate(self):
        # 添加学习率调度
        initial_lr = self.config['training']['learning_rate']
        lr_schedule= tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr,
            decay_steps=self.config['training']['num_iterations'] // 5,
            decay_rate=0.8,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)

    def train(self, evaluation_data=None, training_data=None):
        
        if evaluation_data != None: self.evaluation_data = evaluation_data
        if training_data != None: self.training_data = training_data
        self.policy_model.summary()

        # 训练循环
        for iteration in range(self.config['training']['num_iterations']):

            batch_experience = get_padded_batch_data(self.memory, self.training_data, self.config['training']['batch_size'], 1, self.PAD_ID, self.START_ID, self.END_ID, self.config['predict']['max_length'])
            loss = self._train_step(batch_experience, self.policy_model)

            # 打印进度
            if iteration % self.config['training']['num_print'] == 0:
                print(f"step: {iteration} - loss: {loss:.6f}")
            
            # 评估
            if iteration % self.config['training']['num_show_play'] == 0:
                print("--------------------------------")
                _ret = self.play(batch_size=self.config['training']['batch_size'], run_type="training")
                _rewards = [d['reward'] for d in _ret]
                correct_rate = sum(1 for reward in _rewards if reward == 1) / len(_rewards) if len(_rewards) > 0 else 0
                print(f"- 训练数据 - 正确率: {correct_rate:.3f}")
                print("--------------------------------")
                _ret = self.play(batch_size=self.config['training']['batch_size'], run_type="evaluation")
                _rewards = [d['reward'] for d in _ret]
                correct_rate = sum(1 for reward in _rewards if reward == 1) / len(_rewards) if len(_rewards) > 0 else 0
                print(f"- 评估数据 - 正确率: {correct_rate:.3f}")
                self.play(batch_size=1, is_print=True,  run_type="evaluation")
                print("--------------------------------")

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch_experience, model):
        questions, decoder_inputs, answers, rewards = batch_experience
        mask = tf.cast(tf.not_equal(answers, self.PAD_ID), dtype=tf.float32)
        with tf.GradientTape() as tape:
            logits = model((questions, decoder_inputs), training=True)
            loss = self.loss_object(answers, logits)
            loss *= mask
            loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def predict(self, questions, exploration=False):
        # 确保输入是张量形式
        if not isinstance(questions, tf.Tensor):
            questions = tf.convert_to_tensor(questions, dtype=tf.int64)
            
        batch_size = tf.shape(questions)[0]
        decoder_inputs = tf.ones((batch_size, 1), dtype=tf.int64) * tf.cast(self.START_ID, tf.int64)

        # output 容器
        output = []
        for i in range(batch_size):
            output.append(tf.TensorArray(tf.int64, size=self.config['predict']['max_length']))
        
        # 记录每个样本是否已完成生成
        end_flags = tf.zeros((batch_size,), dtype=tf.bool)
        
        max_length = self.config['predict']['max_length']
        for t in tf.range(max_length):
            # 获取模型输出的logits
            logits = self.policy_model((questions, decoder_inputs), training=False)
            logits = logits[:, -1:, :]  # 只考虑最后一个时间步的预测 (batch_size, 1, vocab_size)
            logits = tf.squeeze(logits, axis=1)  # (batch_size, vocab_size)
            
            scaled_logits = logits / self.current_temperature
            
            if exploration:
                probs = tf.nn.softmax(scaled_logits, axis=-1)
                predicted_token = tf.random.categorical(tf.math.log(probs + 1e-8), num_samples=1)  # (batch_size, 1)
            else:
                predicted_token = tf.expand_dims(tf.argmax(scaled_logits, axis=-1), axis=-1)
            
            # 将预测添加到输出
            for i in range(batch_size):
                if not end_flags[i]:
                    output[i] = output[i].write(t, predicted_token[i, 0])
                else:
                    output[i] = output[i].write(t, self.PAD_ID)  # 已结束的序列填充PAD
            
            # 检查是否有序列结束
            new_end_flags = tf.logical_or(end_flags, tf.squeeze(tf.equal(predicted_token, self.END_ID), axis=-1))
            end_flags = new_end_flags
            
            # 如果所有序列都结束了，提前停止
            if tf.reduce_all(end_flags):
                break
                
            # 更新解码器输入
            decoder_inputs = tf.concat([decoder_inputs, predicted_token], axis=1)
            
            # 如果序列已经足够长，提前停止
            early_stop_threshold = tf.cast(tf.cast(batch_size, tf.float32) * 0.95, tf.int32)
            if t > int(max_length * 0.3) and tf.reduce_sum(tf.cast(end_flags, tf.int32)) > early_stop_threshold:
                break
        
        # output 容器
        final_output = []
        for i in range(batch_size):
            final_output.append(output[i].stack())
        final_output = tf.stack(final_output)  # (batch_size, sequence_length)

        return final_output

    def play(self,batch_size=1,is_print=False, run_type="training", exploration=False):

        # sample questions
        questions = []

        if run_type == "evaluation":
            _data = random.sample(self.evaluation_data, batch_size)
        elif run_type == "training":
            _data = random.sample(self.training_data, batch_size)
        elif run_type == "test":
            _data = self.env.roll_cold_start_data(batch_size)
            
        for q in _data:
            questions.append(q['question'])
        questions_tensor = tf.convert_to_tensor(questions, dtype=tf.int64)

        # predict
        outputs = self.predict(questions_tensor, exploration=exploration)
        processed_outputs = truncate_end(outputs, self.END_ID)
        
        # 环境交互
        rewards = []
        for i in range(len(processed_outputs)):
            if is_print: print("预测tokens",processed_outputs[i])
            reward = self.env.play(state_tokens=questions[i], action_tokens=processed_outputs[i], is_print=is_print)
            rewards.append(reward)
        
        # 组合数据
        data = []
        for i in range(len(processed_outputs)):
            data.append({'question': questions[i], 'answer': processed_outputs[i], 'reward': rewards[i]})

        return data