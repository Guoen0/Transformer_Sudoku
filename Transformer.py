import tensorflow as tf
from tensorflow import keras
from keras import layers

class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate)  
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate)
        self.final_layer = layers.Dense(target_vocab_size)  

    def call(self, inputs, training=False):
        inp, tar = inputs
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        
        enc_output = self.encoder(
            x=inp,
            training=training,
            mask=enc_padding_mask
        )
        
        dec_output = self.decoder(
            x=tar,
            enc_output=enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask
        )
        
        final_output = self.final_layer(dec_output)
        return final_output
    
    def create_masks(self, inp, tar):

        enc_padding_mask = self.create_padding_mask(inp) 
        dec_padding_mask = self.create_padding_mask(inp)

        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])  
        dec_target_padding_mask = self.create_padding_mask(tar)

        #combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        combined_mask = dec_target_padding_mask * look_ahead_mask

        return enc_padding_mask, combined_mask, dec_padding_mask    

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]    

    def create_look_ahead_mask(self, size):
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, pos, i, d_model):
        # 确保所有输入都是float32类型
        pos = tf.cast(pos, tf.float32)
        i = tf.cast(i, tf.float32)
        d_model = tf.cast(d_model, tf.float32)
        
        # 计算角度率
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / d_model)
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position)[:, tf.newaxis],
            tf.range(d_model)[tf.newaxis, :],
            d_model
        )
        
        # 对偶数位置应用sin
        sines = tf.math.sin(angle_rads[:, 0::2])
        # 对奇数位置应用cos
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # 交错合并sin和cos值
        pos_encoding = tf.stack([sines, cosines], axis=2)
        pos_encoding = tf.reshape(pos_encoding, [position, d_model])
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:tf.shape(inputs)[1], :]

class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers        

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, training=False, mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
            
        return x

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)   
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn_output = self.multi_head_attention(
            query=x,
            key=x,
            value=x,
            attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)   

        return self.layernorm2(out1 + ffn_output)
    
class point_wise_feed_forward_network(layers.Layer):
    def __init__(self, d_model, dff):
        super(point_wise_feed_forward_network, self).__init__()

        self.d_model = d_model
        self.dff = dff

        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))
    
    
class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )
        
        return x

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.multi_head_attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)  
        self.multi_head_attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)  
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)   

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):

        attn_output1 = self.multi_head_attention1(x, x, x, look_ahead_mask)

        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(x + attn_output1)

        attn_output2 = self.multi_head_attention2(
            query = out1,
            key = enc_output,
            value = enc_output,
            attention_mask = padding_mask
        )

        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(out1 + attn_output2)

        ffn_output = self.ffn(out2)

        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3   