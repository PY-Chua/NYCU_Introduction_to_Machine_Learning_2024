import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

data = pd.read_csv('HW1.csv')

# Extract features and target
#X = data.iloc[:, 1:].values  # Exclude the first column (song_popularity)
x = data.drop('song_popularity', axis = 1)
#x.insert(0, 'bias', [1]*len(x))
X = x.values
y = data['song_popularity'].values

# Split the data into training and testing sets
x_train, x_test = X[:10000], X[10000:]
y_train, y_test = y[:10000], y[10000:]
# Define basis function parameters
M_values = [1, 3, 5, 10, 20, 30]
s = 0.1
M = max(M_values)
µ = np.array([3*j/M for j in range(1, M)])

# Define logistic sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Define basis functions
def basis_functions(x, M, s, µ):
    phi = np.zeros((len(x), M))
    phi[:, 0] = 1
    for j in range(1, M):
        phi[:, j] = sigmoid((x - µ[j-1]) / s)
    return phi

# Define function for fitting curve
def fit_curve(x, y, M_values, s, µ):
    fig, axs = plt.subplots(len(M_values), 1, figsize=(10, 6*len(M_values)))
    for i, M in enumerate(M_values):
        phi = basis_functions(x, M, s, µ)
        model = LinearRegression().fit(phi, y)
        y_pred = model.predict(phi)
        axs[i].scatter(x, y, color='blue', label='Original data')
        axs[i].plot(x, y_pred, color='red', label='Fitted curve')
        axs[i].set_title(f'Fitting curve for M = {M}')
        axs[i].legend()
    plt.tight_layout()
    plt.show()

# Part I-1: Plotting fitting curve for different M values
x = x_train[:, 2]
fit_curve(x, y_train, M_values, s, µ)

# Part I-2: Plotting Mean Square Error and accuracy for different M values
def calculate_metrics(X, y, M_values, s, µ, regularization=False, λ=None):
    mse_train = []
    mape_train = []
    mse_test = []
    mape_test = []
    for M in M_values:
        phi_train = basis_functions(X, M, s, µ)
        phi_test = basis_functions(X_test['x3'].values.reshape(-1, 1), M, s, µ)
        
        if regularization:
            model = Ridge(alpha=λ)
        else:
            model = LinearRegression()
        
        model.fit(phi_train, y)
        
        y_train_pred = model.predict(phi_train)
        mse_train.append(mean_squared_error(y, y_train_pred))
        mape_train.append(mean_absolute_percentage_error(y, y_train_pred))
        
        y_test_pred = model.predict(phi_test)
        mse_test.append(mean_squared_error(y_test, y_test_pred))
        mape_test.append(mean_absolute_percentage_error(y_test, y_test_pred))
    
    return mse_train, mape_train, mse_test, mape_test

mse_train, mape_train, mse_test, mape_test = calculate_metrics(X_train['x3'].values.reshape(-1, 1), y_train, M_values, s, µ)
plt.figure(figsize=(10, 6))
plt.plot(M_values, mse_train, label='Training MSE')
plt.plot(M_values, mse_test, label='Testing MSE')
plt.xlabel('M')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error for Different M values')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(M_values, mape_train, label='Training Accuracy')
plt.plot(M_values, mape_test, label='Testing Accuracy')
plt.xlabel('M')
plt.ylabel('Mean Absolute Percentage Error')
plt.title('Mean Absolute Percentage Error for Different M values')
plt.legend()
plt.show()

# Part I-3: Cross-validation to select the best order M
def cross_validation(X, y, M_values, s, µ, folds=5):
    kf = KFold(n_splits=folds)
    best_M = None
    best_mse = float('inf')
    for M in M_values:
        mse_cv = []
        phi = basis_functions(X, M, s, µ)
        for train_index, val_index in kf.split(X):
            X_train_cv, X_val = X[train_index], X[val_index]
            y_train_cv, y_val = y[train_index], y[val_index]
            model = LinearRegression().fit(phi[train_index], y_train_cv)
            y_val_pred = model.predict(phi[val_index])
            mse_cv.append(mean_squared_error(y_val, y_val_pred))
        avg_mse_cv = np.mean(mse_cv)
        if avg_mse_cv < best_mse:
            best_mse = avg_mse_cv
            best_M = M
    return best_M

best_M = cross_validation(X_train['x3'].values.reshape(-1, 1), y_train, M_values, s, µ)
print(f'Best order M selected through cross-validation: {best_M}')

# Part I-4: Repeat Part I-1 and Part I-2 with regularization
λ_values = [0.1, 1, 10]
for λ in λ_values:
    mse_train_reg, mape_train_reg, mse_test_reg, mape_test_reg = calculate_metrics(X_train['x3'].values.reshape(-1, 1), y_train, M_values, s, µ, regularization=True, λ=λ)
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, mse_train_reg, label='Training MSE')
    plt.plot(M_values, mse_test_reg, label='Testing MSE')
    plt.xlabel('M')
    plt.ylabel('Mean Square Error')
    plt.title(f'Mean Square Error with Regularization (λ = {λ})')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(M_values, mape_train_reg, label='Training Accuracy')
    plt.plot(M_values, mape_test_reg, label='Testing Accuracy')
    plt.xlabel('M')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.title(f'Mean Absolute Percentage Error with Regularization (λ = {λ})')
    plt.legend()
    plt.show()
