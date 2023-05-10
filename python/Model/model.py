from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model, losses
from collections import defaultdict

class Contrastive_Model(keras.Model):
    def __init__(self, train_flow, temperature, a0, a1, inputs, outputs, label_smoothing):
        super().__init__()
        self.train_flow = train_flow
        self.temperature = temperature
        self.a0 = a0
        self.a1 = a1
        self.model = Model(inputs = inputs, outputs = outputs)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits = False, label_smoothing = label_smoothing)


    def call(self, data):
        return self.model(data)

        
    def train_step(self, data):
        x, y = data
        sample_weight = None

        with tf.GradientTape() as tape:
            # Forward
            x_out = self(x, training=True)
            f1, f2 = x_out[0], x_out[1]
            y_pred = tf.sigmoid(tf.reduce_sum(f1 * f2, axis = 1))
            
            # Loss
            axis0_loss, axis1_loss = self.get_loss(f1, f2)
            axis0_loss = self.a0 * axis0_loss
            axis1_loss = self.a1 * axis1_loss
            bce_loss = self.bce(y, y_pred)
            #self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = bce_loss + axis0_loss + axis1_loss

            # Acc
            y = tf.cast(y, dtype = tf.float32)
            acc = tf.reduce_mean(tf.cast(tf.where(tf.greater(y_pred, 0.5), 1.0,0.0) == y, dtype = tf.float32))

            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        result = dict()
        result["Total_loss"] = loss
        result["Contrastive_loss(axis0)"] = axis0_loss
        result["Contrastive_loss(axis1)"] = axis1_loss
        result["BCE_loss"] = bce_loss
        result["Train_acc"] = acc
        return result
    
    def test_step(self, data):
        # forward
        x, y = data
        x_out = self(x, training=True)
        f1, f2 = x_out[0], x_out[1]            
        y_pred = tf.sigmoid(tf.reduce_sum(f1 * f2, axis = 1))

        y = tf.cast(y, dtype = tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.where(tf.greater(y_pred, 0.5), 1.0, 0.0) == y, dtype = tf.float32))
        bce_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # save
        result = {}
        result["Acc"] = acc
        result["Loss"] = bce_loss
        return result
    
        
    def get_loss(self, f1, f2):

        train_flow = self.train_flow
        temperature = self.temperature
        
        csv_dict = defaultdict(list)
        wiki_dict = defaultdict(list)
        csv_wiki_map = []
        csv_id_mapping = {}
        wiki_id_mapping = {}
        csv_wiki_id_matching = []

        for idx, (csv_id, wiki_id) in enumerate(train_flow.ids):
            csv_dict[csv_id].append(f1[idx])
            wiki_dict[wiki_id].append(f2[idx])

            if csv_id not in csv_id_mapping.keys():
                csv_id_mapping[csv_id] = len(csv_id_mapping)

            if wiki_id not in wiki_id_mapping.keys():
                wiki_id_mapping[wiki_id] = len(wiki_id_mapping)

            mapping_id_csv = csv_id_mapping[csv_id]
            mapping_id_wiki = wiki_id_mapping[wiki_id]

            csv_wiki_id_matching.append((mapping_id_csv, mapping_id_wiki))

        csv_wiki_map = [[0] * len(wiki_dict) for _ in range(len(csv_dict))]

        for row,col in csv_wiki_id_matching:
            csv_wiki_map[row][col] = 1

        csv_wiki_map = tf.convert_to_tensor(csv_wiki_map)

        csv_embeddings = []
        for value in list(csv_dict.values()):
            csv_embeddings.append(tf.math.reduce_mean(tf.convert_to_tensor(value), axis = 0))
        csv_embeddings = tf.convert_to_tensor(csv_embeddings)

        wiki_embeddings = []
        for value in list(wiki_dict.values()):
            wiki_embeddings.append(tf.math.reduce_mean(tf.convert_to_tensor(value), axis = 0))
        wiki_embeddings = tf.convert_to_tensor(wiki_embeddings)

        mm_result = tf.divide(tf.matmul(csv_embeddings, tf.transpose(wiki_embeddings)), temperature)

        exp_mm_result = tf.exp(mm_result)

        sum_mm_result = tf.math.reduce_sum(exp_mm_result, axis = 1)
        normalized_result = tf.math.log(tf.divide(exp_mm_result, tf.expand_dims(sum_mm_result, axis = 1)))
        csv_wiki_map = tf.cast(csv_wiki_map, dtype = tf.float32)
        masked_result = tf.math.multiply(normalized_result, csv_wiki_map)
        axis1_loss = -tf.reduce_mean(tf.reduce_sum(masked_result, axis = 1) / tf.reduce_sum(csv_wiki_map, axis = 1))

        sum_mm_result = tf.math.reduce_sum(exp_mm_result, axis = 0)
        normalized_result = tf.math.log(tf.divide(exp_mm_result, tf.expand_dims(sum_mm_result, axis = 0)))
        masked_result = tf.math.multiply(normalized_result, csv_wiki_map)
        axis0_loss = -tf.reduce_mean(tf.reduce_sum(masked_result, axis = 0) / tf.reduce_sum(csv_wiki_map, axis = 0))
        
        return (axis0_loss, axis1_loss)