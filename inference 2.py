import numpy as np
import tensorflow as tf

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_input(input_data, expected_shape):
    # Resize if needed (for safety)
    if input_data.shape != tuple(expected_shape[1:]):
        input_data = tf.image.resize(input_data, expected_shape[1:3]).numpy()
    
    # Normalize input data to [0,1] range or as needed
    input_data = input_data.astype(np.float32) / 255.0

    # Expand dims if needed
    if len(input_data.shape) == len(expected_shape) - 1:
        input_data = np.expand_dims(input_data, axis=0)

    return input_data

def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_data = preprocess_input(input_data, input_shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# --- Load interpreters for each stream ---
rgb_model_path = "tflite_models/best_models/best_rgb_feature_model_quantized.tflite"
motion_model_path = "tflite_models/best_models/best_motion_feature_model_quantized.tflite"
lstm_model_path = "tflite_models/best_models/best_combined_lstm_classifier_quantized.tflite"

rgb_interpreter = load_tflite_model(rgb_model_path)
motion_interpreter = load_tflite_model(motion_model_path)
lstm_interpreter = load_tflite_model(lstm_model_path)

# --- Simulated Inputs (replace with real video frames/slices) ---
# Assume 10 time steps (frames) per window, 224x224 resolution, 3 channels (RGB)
sequence_length = 10
frame_height, frame_width, channels = 224, 224, 3

rgb_sequence = np.random.rand(sequence_length, frame_height, frame_width, channels)
motion_sequence = np.random.rand(sequence_length, frame_height, frame_width, 1)  # grayscale or optical flow

# --- Extract features per frame (RGB & motion streams) ---
rgb_features = []
motion_features = []

for i in range(sequence_length):
    rgb_feat = run_tflite_inference(rgb_interpreter, rgb_sequence[i])
    motion_feat = run_tflite_inference(motion_interpreter, motion_sequence[i])
    rgb_features.append(rgb_feat)
    motion_features.append(motion_feat)

rgb_features = np.array(rgb_features)
motion_features = np.array(motion_features)

# --- Combine features ---
# You might concatenate or fuse features based on your training architecture
# Example: concatenate along feature dimension
combined_features = np.concatenate([rgb_features, motion_features], axis=-1)
combined_features = np.expand_dims(combined_features, axis=0)  # shape: (1, seq_len, feature_dim)

# --- Run final LSTM + Dense inference ---
input_details = lstm_interpreter.get_input_details()
output_details = lstm_interpreter.get_output_details()

# Handle quantization if needed
if input_details[0]['dtype'] == np.uint8:
    scale, zero_point = input_details[0]['quantization']
    combined_features = combined_features / scale + zero_point
    combined_features = combined_features.astype(np.uint8)

lstm_interpreter.set_tensor(input_details[0]['index'], combined_features)
lstm_interpreter.invoke()
final_output = lstm_interpreter.get_tensor(output_details[0]['index'])

# --- Print prediction ---
print("Final Prediction:", final_output)
