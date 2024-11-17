
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
import os

# Simulating a dataset (replace with your actual dataset)
# Replace X and y with your actual dataset variables
np.random.seed(42)
X = np.random.rand(1000, 10)  # Example features
y = np.random.rand(1000)     # Example target

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * 0.9

# Define the updated model
def updated_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='linear'))  # Regression output
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Instantiate and compile the model
input_dim = X_train.shape[1]
updated_ann_model = updated_model(input_dim)

# Learning rate scheduler
lr_scheduler = LearningRateScheduler(scheduler)

# Train the updated model
history_updated = updated_ann_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[lr_scheduler],
    verbose=1
)

# Evaluate the model
results = updated_ann_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {results[0]:.4f}, Test MAE: {results[1]:.4f}")

# Save the updated model
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')

updated_ann_model.save('trained_models/ANN_updated_model.keras')

# Save predictions to CSV
y_pred = updated_ann_model.predict(X_test)
output_dir = 'predictions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pd.DataFrame(zip(y_test, y_pred.flatten()), columns=['test', 'pred']).to_csv(
    f'{output_dir}/pred_ANN_updated_model.csv', index=False
)
