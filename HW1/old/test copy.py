import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def logistic_sigmoid(a):
    return 1 / (1 + np.exp(-a))

def basis_function(x, M):
    s = 0.1
    mu = np.array([3 * j / M for j in range(1, M)])
    phi = np.array([logistic_sigmoid((x - mu[j]) / s) if j != 0 else np.ones(len(x)) for j in range(M)])
    return phi
'''
def compute_concatenated_basis_functions(x_data, M):
    phi_all = []
    for index in range(x_data.shape[1]):  # Assuming x_data is a 2D array where each column represents an input feature
        # Compute basis functions for each input feature separately
        phi = basis_function(x_data[:, index], M)
        phi_all.append(phi)
    # Concatenate the basis functions for all input features horizontally
    return np.column_stack(phi_all)
'''
def predict(x, w):
    return np.dot(x, w)

def wml(phi,t):
    C = np.linalg.pinv(np.dot(phi.T, phi))
    beta = np.dot(np.dot(C, phi.T), t)
    return beta

def calculate_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def mean_absolute_percentage_error(y, t):
    return np.mean(np.abs((y - t) / np.maximum(np.abs(t), 1))) * 100


# Load data
data = pd.read_csv('HW1.csv')

# Extract features and target
X = data.iloc[:, 1:].values.reshape(-1, 1)  # Exclude the first column (song_popularity)
t = data['song_popularity'].values.reshape(-1, 1)

# Split data into training and testing sets
x_train, x_test = X[:10000], X[10000:]
t_train, t_test = t[:10000], t[10000:]

# Part 1 - Plot fitting curve for the third input feature (x3: danceability) for various M
M_values = [1, 3, 5, 10, 20, 30]
feature_index = 2  # Indexing starts from 0, so x3 is at index 2

plt.figure(figsize=(12, 6))
for M in M_values:
    phi_train = basis_function(x_train, M)
    wml_result = wml(phi_train, t_train)
    #x_predict = np.linspace(0, 3, 100)
    phi_predict = basis_function(x_test, M)
    y_predict = predict(phi_predict, wml_result)
    plt.plot(x_test, y_predict, label=f'M={M}')
    plt.show()

plt.scatter(x_train[:, feature_index], t_train, color='blue', label='Training data')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.title('Fitting Curve for Danceability with Various M')
plt.legend()
plt.grid(True)
plt.show()
