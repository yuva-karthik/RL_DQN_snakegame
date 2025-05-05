import tensorflow as tf
from tensorflow.keras import layers, models

class DQNModel(models.Model):
    def __init__(self):
        super(DQNModel,self).__init__()
        self.dense1 = layers.Dense(128,activation='relu',input_shape=(11,))
        self.dense2 = layers.Dense(64,activation='relu')
        self.output_layer = layers.Dense(3)
    
    def call(self,inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
