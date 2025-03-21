import random
import tensorflow as tf
import yaml

def get_padded_batch_data(data_1,data_2,batch_size,data_ratio,PAD_ID,START_ID,END_ID,max_length=500):
    # 采样1批数据
    data_2_size = int(batch_size * data_ratio)
    data_1_size = batch_size - data_2_size
    
    # 确保不超过可用数据量
    data_2_size = min(data_2_size, len(data_2))
    data_1_size = min(data_1_size, len(data_1))
    
    # 安全采样
    data_2_sample = random.sample(data_2, data_2_size) if data_2_size > 0 else []
    data_1_sample = random.sample(data_1, data_1_size) if data_1_size > 0 else []
    data_1_sample = data_1_sample + data_2_sample

    encoder_inputs = [record["question"] for record in data_1_sample]

    # 处理序列 - 单独处理每个样本
    targets = [record["answer"] for record in data_1_sample]
    real_labels = []
    decoder_inputs = []
    for target in targets:
        # 限制目标序列长度，确保不超过最大长度减去END_ID
        truncated_target = target[:max_length-1] if len(target) > max_length-1 else target
        real_labels.append(list(truncated_target) + [END_ID])
        decoder_inputs.append([START_ID] + list(truncated_target))

    # RL
    rewards = [record["reward"] for record in data_1_sample]

    # 限制输入序列长度
    encoder_inputs = [seq[:max_length] if len(seq) > max_length else seq for seq in encoder_inputs]

    # <PAD>填充序列
    padded_encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoder_inputs, padding='post', value=PAD_ID)
    padded_decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_inputs, padding='post', value=PAD_ID)
    padded_real_labels = tf.keras.preprocessing.sequence.pad_sequences(real_labels, padding='post', value=PAD_ID)
    
    # 类型转换
    padded_encoder_inputs = tf.cast(padded_encoder_inputs, tf.int64)
    padded_decoder_inputs = tf.cast(padded_decoder_inputs, tf.int64)
    padded_real_labels = tf.cast(padded_real_labels, tf.int64)
    rewards = tf.cast(rewards, tf.float32)
    
    return ([
        padded_encoder_inputs,
        padded_decoder_inputs,
        padded_real_labels,
        rewards
    ])

def truncate_end(outputs,END_ID):
    processed_outputs = []
    for i in range(len(outputs)):
        # 将张量转换为NumPy数组
        output_np = outputs[i].numpy()
        # 找到<END>标记的位置并截断
        end_idx = tf.where(outputs[i] == END_ID)
        if tf.size(end_idx) > 0:  # 如果找到了<END>标记
            end_pos = end_idx[0][0].numpy()
            output_np = output_np[:end_pos]
        processed_outputs.append(output_np)
    return processed_outputs

def get_indices(real_labels):
    batch_size = tf.shape(real_labels)[0]
    seq_length = tf.shape(real_labels)[1]
    
    # 创建批次索引 [0,0,0,...,1,1,1,...,2,2,2,...]
    batch_indices = tf.repeat(tf.range(batch_size), seq_length)
    batch_indices = tf.reshape(batch_indices, [-1, 1])
    
    # 创建序列索引 [0,1,2,...,0,1,2,...,0,1,2,...]
    seq_indices = tf.tile(tf.range(seq_length), [batch_size])
    seq_indices = tf.reshape(seq_indices, [-1, 1])
    
    # 创建标记索引
    token_indices = tf.reshape(real_labels, [-1, 1])
    token_indices = tf.cast(token_indices, tf.int32)
    
    # 确保所有索引具有相同的第一维大小
    indices = tf.concat([batch_indices, seq_indices, token_indices], axis=1)
    return indices

def loss_mask(answers, END_ID):
    # 创建与 answers 相同形状的初始 mask
    mask = tf.ones(tf.shape(answers), dtype=tf.int32)  # 初始化 mask，长度与 answers 相同

    # 找到每个批次中第一个等于 END_ID 的位置
    end_indices = tf.cast(tf.argmax(tf.cast(tf.equal(answers, END_ID), tf.int32), axis=1), tf.int32)

    # 检查每个批次中是否存在 END_ID
    exists_end_ids = tf.reduce_any(tf.equal(answers, END_ID), axis=1)  # 检查是否存在

    # 生成一个布尔掩码，表示从 end_index 开始的位置为 True
    zero_masks = tf.where(
        exists_end_ids[:, tf.newaxis],  # 扩展维度以匹配
        tf.range(tf.shape(answers)[1]) > tf.expand_dims(end_indices, 1),  # 生成布尔掩码
        tf.zeros_like(mask, dtype=tf.bool)  # 如果不存在，返回全 False
    )

    # 更新 mask
    updated_masks = tf.where(zero_masks, tf.zeros_like(mask), mask)
    updated_masks = tf.cast(updated_masks, dtype=tf.float32)

    return updated_masks

def load_config(config_path="models/config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")

def save_config(agent, config_path="models/config.yaml"):
    config = agent.config

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
