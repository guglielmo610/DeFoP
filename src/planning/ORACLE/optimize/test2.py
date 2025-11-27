# debug_model_loading.py
import tensorflow as tf
import h5py
import numpy as np

def analyze_model_file(checkpoint_path):
    print(f"=== Analyzing: {checkpoint_path} ===")
    
    # Method 1: Try direct loading
    try:
        model = tf.keras.models.load_model(checkpoint_path)
        print("✓ Direct load successful")
        return model
    except Exception as e:
        print(f"✗ Direct load failed: {e}")
    
    # Method 2: Analyze HDF5 structure
    try:
        with h5py.File(checkpoint_path, 'r') as f:
            print("\nHDF5 structure:")
            
            def print_info(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 {name}: shape={obj.shape}, dtype={obj.dtype}")
            
            f.visititems(print_info)
            
            # Check for model config
            if 'model_config' in f.attrs:
                import json
                config = json.loads(f.attrs['model_config'])
                print(f"\nModel class: {config.get('class_name', 'Unknown')}")
                
    except Exception as e:
        print(f"Error analyzing HDF5: {e}")
    
    return None

# Test the problematic file
checkpoint_path = "/home/nvidia/norv_vae/src/planning/ORACLE/models/4/saved-model-20.hdf5"
analyze_model_file(checkpoint_path)