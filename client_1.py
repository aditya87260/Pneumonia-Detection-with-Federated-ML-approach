import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.callbacks import ReduceLROnPlateau

import os

# Define the paths for Client 1
train_dir = "client_1"
test_dir = "/Users/adityasrivastava/Documents/xray_ml/chest_xray/chest_xray/test"
val_dir = "/Users/adityasrivastava/Documents/xray_ml/chest_xray_chest_xray/val"

num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))

weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)
weight_for_1 = num_normal / (num_normal + num_pneumonia)

class_weight = {0: weight_for_0, 1: weight_for_1}



print(f"Weight for class 0: {weight_for_0:.2f}")
print(f"Weight for class 1: {weight_for_1:.2f}")



# ImageDataGenerator for training and validation data
image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

train = image_generator.flow_from_directory(train_dir, 
                                            batch_size=16, 
                                            shuffle=True, 
                                            class_mode='binary',
                                            target_size=(180,180),
                                           classes=['NORMAL', 'PNEUMONIA'])

validation = image_generator.flow_from_directory(val_dir, 
                                                batch_size=1, 
                                                shuffle=False, 
                                                class_mode='binary',
                                                target_size=(180, 180),
                                                classes=['NORMAL', 'PNEUMONIA'])

test = image_generator.flow_from_directory(test_dir, 
                                            batch_size=1, 
                                            shuffle= False, 
                                            class_mode='binary',
                                            target_size=(180, 180),
                                            classes=['NORMAL', 'PNEUMONIA'])


# Build the model
def create_model():
    base_model = inception_v3.InceptionV3(include_top= False, weights= "imagenet", input_shape=(180,180, 3), pooling= 'max')
    
    base_model.trainable = False

    # Freeze all layers except for the last 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    inputs = layers.Input(shape=(180,180, 3))
    
    x = base_model(inputs)

    # Head
    #reshape the output of the base model to ndim =4
    x = layers.Reshape((1, 1, 2048))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    #Final Layer (Output)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=[inputs], outputs=output)
    model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics=['accuracy'])
    return model

class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data, test_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def get_parameters(self,config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        callback_lr = ReduceLROnPlateau(monitor='val_loss', patience = 3, cooldown=0, verbose=1, factor=0.6, min_lr=0.000001)
        r = self.model.fit(self.train_data, epochs=1, validation_data=self.val_data, class_weight={0:12, 1:0.5}, callbacks=[callback_lr])
        hist = r.history
        print("Fit history : " ,hist)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data)
        print("Eval accuracy : ", accuracy)
        return loss, len(self.test_data), {"accuracy": accuracy}

# Start the client
if __name__ == "__main__":
    model = create_model()
    client = PneumoniaClient(model, train, validation, test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",grpc_max_message_length=1024 * 1024 * 1024, client=client)
    
