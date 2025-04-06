import numpy as np
import tensorflow as tf

# Load models
rgb_model = tf.keras.models.load_model('final_models/rgb_features_model.keras')
motion_model = tf.keras.models.load_model('final_models/motion_features_model.keras')
se_model = tf.keras.models.load_model('final_models/se_block_model.keras')
dense_model = tf.keras.models.load_model('final_models/dense_frozen_model.keras')

# Dummy input data (replace with actual preprocessed video clip)
time_steps = 16
rgb_input = np.random.rand(1, time_steps, 224, 224, 3).astype(np.float32)
motion_input = np.random.rand(1, time_steps, 224, 224, 2).astype(np.float32)

# Step 1: Get features from both streams
rgb_features = rgb_model(rgb_input)               # shape: (1, T, feature_dim)
motion_features = motion_model(motion_input)      # shape: (1, T, feature_dim)

print("RGB Features shape:", rgb_features.shape)
print("Motion Features shape:", motion_features.shape)

# Step 2: Concatenate along channel dimension
combined_features = tf.concat([rgb_features, motion_features], axis=-1)  # shape: (1, T, combined_dim)

# Step 3: Apply SE block (frame-wise reweighting)
se_features = se_model(combined_features)  # shape: (1, T, se_dim)
print("SE Features shape:", se_features.shape)

# Step 4: Apply Dense classifier to each frame/timestep
# We can flatten the batch and time steps before classification
batch_size, time_steps, feature_dim = se_features.shape
se_flattened = tf.reshape(se_features, (-1, feature_dim))  # shape: (T, feature_dim)

# Step 5: Frame-wise classification
frame_predictions = dense_model(se_flattened)              # shape: (T, num_classes)

# Optional: aggregate predictions across time (e.g., by averaging)
final_prediction = tf.reduce_mean(frame_predictions, axis=0)
predicted_class = tf.argmax(final_prediction).numpy()

# Print results
print("\nFrame-wise Predictions:\n", frame_predictions.numpy())
print("\nFinal Averaged Prediction:\n", final_prediction.numpy())
print("Predicted Class Index:", predicted_class)
