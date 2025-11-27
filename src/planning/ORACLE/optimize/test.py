# simple_convert.py
import tensorflow as tf
import numpy as np

def simple_model_convert(checkpoint_path):
    """Simple conversion approach"""
    
    # Create a simple model with the expected architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(34, 60, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), 
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    print("Model created")
    model.build((None, 34, 60, 1))
    
    # Try different loading approaches
    try:
        # Approach 1: Direct load
        model = tf.keras.models.load_model(checkpoint_path)
        print("✓ Model loaded directly")
    except:
        try:
            # Approach 2: Load weights only
            model.load_weights(checkpoint_path)
            print("✓ Weights loaded")
        except:
            # Approach 3: Skip weights and use random initialization
            print("⚠ Using random weights (model will need retraining)")
    
    return model

# Test the simple approach
model = simple_model_convert("/home/nvidia/norv_vae/src/planning/ORACLE/models/4/saved-model-20.hdf5")