import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#add parse arguments
parser= argparse.ArgumentParser(description="hyperpamaterers: on will process hyperparameters.\noff will take already processed parameters.")
parser.add_argument('--hyperparameters', choices=['on', 'off'], default="on", help='Specify whether calculating hyperparameters should be enabled or disabled.')
args = parser.parse_args()

    
#function to add gaussian noise to the images
def add_noise(images, noise_level):
    noisy_images = images + (noise_level * np.random.normal(scale=0.9, size=images.shape)) #add noise
    noisy_images = np.clip(noisy_images, 0., 1.) #clip the values in between 0-1.
    return noisy_images

#load dataset and split it into train, test and validation sets
digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, stratify=digits.target)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)

scaler = MinMaxScaler()
x_train_fit = scaler.fit_transform(x_train)
x_val_fit = scaler.transform(x_val)
x_test_fit = scaler.transform(x_test)

#noise the datas
x_train_noisy_low = add_noise(x_train_fit, 0.1) 
x_train_noisy_medium = add_noise(x_train_fit, 0.3)
x_train_noisy_high = add_noise(x_train_fit, 0.5)

x_val_noisy_low = add_noise(x_val_fit, 0.1)
x_val_noisy_medium = add_noise(x_val_fit, 0.3)
x_val_noisy_high = add_noise(x_val_fit, 0.5)

x_test_noisy_low = add_noise(x_test_fit, 0.1)
x_test_noisy_medium = add_noise(x_test_fit, 0.3)
x_test_noisy_high = add_noise(x_test_fit, 0.5)


#define parameters
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (256,)],
    'activation': ['relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant'],
}

#apply grid search
if args.hyperparameters == "on":
    grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=3)
    grid_search.fit(x_train_noisy_medium, x_train_fit)

    best_params = grid_search.best_params_
    print(best_params)

    with open('best_params.json', 'w') as json_file:
        json.dump(best_params, json_file)
        
if args.hyperparameters == "off":
    with open('best_params.json', 'r') as json_file:
        best_params = json.load(json_file)

#construct auto enoder 
ae_model = MLPRegressor(**best_params, max_iter=1000)

#train the model and get MSE values for both sets
train_mse_values = []
val_mse_values = []

for epoch in range(500):
    ae_model.partial_fit(x_train_noisy_medium, x_train_fit)
    train_mse = mean_squared_error(x_train_fit, ae_model.predict(x_train_noisy_medium))
    val_mse = mean_squared_error(x_val_fit, ae_model.predict(x_val_noisy_medium))
    train_mse_values.append(train_mse)
    val_mse_values.append(val_mse)

#plot the error curves
plt.plot(train_mse_values, label='Train MSE')
plt.plot(val_mse_values, label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Mean Squared Error Curves')
plt.legend()
plt.show()

#print MSE of test set
test_mse_low = mean_squared_error(x_test_fit, ae_model.predict(x_test_noisy_low))
test_mse_medium = mean_squared_error(x_test_fit, ae_model.predict(x_test_noisy_medium))
test_mse_high = mean_squared_error(x_test_fit, ae_model.predict(x_test_noisy_high))

print("Mean Squared Error of Test Set (Low Noise):", test_mse_low)
print("Mean Squared Error of Test Set (Medium Noise):", test_mse_medium)
print("Mean Squared Error of Test Set (High Noise):", test_mse_high)

#show original, noised, and denoised images (randomly selected)
selected_indices = np.random.choice(len(x_test_fit), size=4, replace=False)

fig, axs = plt.subplots(4, 3, figsize=(10, 12))

for i, index in enumerate(selected_indices):
    axs[i, 0].imshow(x_test_fit[index].reshape(8, 8), cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(x_test_noisy_medium[index].reshape(8, 8), cmap='gray')
    axs[i, 1].set_title('Noised (Medium)')
    axs[i, 1].axis('off')

    axs[i, 2].imshow(ae_model.predict(x_test_noisy_medium)[index].reshape(8, 8), cmap='gray')
    axs[i, 2].set_title('Denoised')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()