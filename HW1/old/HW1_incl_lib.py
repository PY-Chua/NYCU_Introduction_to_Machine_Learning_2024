import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

def logistic_sigmoid(a):
    return 1 / (1 + np.exp(-a))

def basis_function(x, m):
    
    s = 0.1
    mu = np.array([3 * j / M for j in range(1, m)])
    #phi = np.ones((len(x), M))
    #for j in range(1, M):
    #    phi[:, j] = logistic_sigmoid((x - mu[j]) / s)
    phi = np.array([logistic_sigmoid((x - mu[j]) / s) if j != 0 else np.ones(len(x)) for j in range(m)])
    return phi.T
 
def compute_concatenated_basis_functions(x_data, M):
    phi_all = []
    for index in range(x_data.shape[1]):  # Assuming x_data is a 2D array where each column represents an input feature
        # Compute basis functions for each input feature separately
        phi = basis_function(x_data[:, index], M)
        phi_all.append(phi)
    # Concatenate the basis functions for all input features horizontally
    return np.column_stack(phi_all)

def predict(x, w):
    return np.dot(x, w)

def wml(phi,t):
    #beta = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), t)
    #wml_result=np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(t)
    #print(wml_result)
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
X = data.iloc[:, 1:].values  # Exclude the first column (song_popularity)
#x = data.drop('song_popularity', axis = 1)
#x.insert(0, 'bias', [1]*len(x))
#X = x.values
t = data['song_popularity'].values

# Split data into training and testing sets
x_train, x_test = X[:10000], X[10000:]
t_train, t_test = t[:10000], t[10000:]

# Part 1 - Plot fitting curve for the third input feature (x3: danceability) for various M
M_values = [[1,3,5],[10,20,30]]
feature_index = 2  # Indexing starts from 0, so x3 is at index 2

fig, ax = plt.subplots(2,3,constrained_layout = True)
fig.suptitle('Part I. (1) for Wml')
for i in range(2):
    for j in range(3):
        m = M_values[i][j]
        #phi_cal = basis_function(x_train, m)
        #wml_cal = wml(phi_cal, t_train)
        #x_predict=np.linspace(0,3,100)
        #y_predict=predict(basis_function(x_predict, m),wml_cal)
        phi_train = basis_function(x_train, m)
        wml_result = wml(phi_train, t_train)
        #x_predict = np.linspace(0, 3, 100)
        phi_predict = basis_function(x_test, m)
        y_predict = predict(phi_predict, wml_result)

        title='M = '+str(m)
        ax[i,j].set_title(title)
        ax[i,j].scatter(x_train[:, 2],t_train,c='b')

        model=interp1d(x_train, y_predict,kind="cubic",fill_value="extrapolate")
        xs=np.linspace(0,3,100)
        ys=model(xs)
        ax[i,j].plot(xs,ys,c='r')
plt.show()

'''
# Part 2 - Plot Mean Square Error and evaluate accuracy for different M values
M_range = range(1, 31)
train_errors = []
test_errors = []
train_accuracies = []
test_accuracies = []

for M in M_range:
    # Compute concatenated basis functions for training and testing data
    phi_train_concat = compute_concatenated_basis_functions(x_train, M)
    phi_test_concat = compute_concatenated_basis_functions(x_test, M)

    # Train the model
    w = np.linalg.lstsq(phi_train_concat, t_train, rcond=None)[0]

    # Predict on training and testing sets
    y_train = predict(phi_train_concat, w)
    y_test = predict(phi_test_concat, w)

    # Calculate errors
    train_error = calculate_error(y_train, t_train)
    test_error = calculate_error(y_test, t_test)

    # Calculate accuracies
    train_accuracy = mean_absolute_percentage_error(y_train, t_train)
    test_accuracy = mean_absolute_percentage_error(y_test, t_test)

    train_errors.append(train_error)
    test_errors.append(test_error)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
plt.figure(figsize=(10, 6))
plt.plot(M_range, train_errors, label='Training Error')
plt.plot(M_range, test_errors, label='Testing Error')
plt.xlabel('M')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error on Training and Testing Sets for Different M')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(M_range, train_accuracies, label='Training Accuracy')
plt.plot(M_range, test_accuracies, label='Testing Accuracy')
plt.xlabel('M')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy on Training and Testing Sets for Different M')
plt.legend()
plt.grid(True)
plt.show()

# Part 3 - Cross-validation to select the best order M
kfolds = 5
best_M = None
best_error = float('inf')

for M in M_range:
    error = cross_validation(x_train, t_train, kfolds, M)
    if error < best_error:
        best_error = error
        best_M = M

print("Best M selected by cross-validation:", best_M)

# Evaluate on testing set with best M
phi_train = basis_function(x_train, best_M)
phi_test = basis_function(x_test, best_M)

# Train the model
w = np.linalg.lstsq(phi_train, t_train, rcond=None)[0]

# Predict on testing set
y_test = predict(phi_test, w)

# Plot fitting curve for danceability
x_range = np.linspace(np.min(x_test[:, feature_index]), np.max(x_test[:, feature_index]), 1000)
phi_range = basis_function(x_range, best_M)
y = predict(phi_range, w)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y, label=f'Best M={best_M}')
plt.scatter(x_test[:, feature_index], t_test, color='red', label='Testing data')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.title('Fitting Curve for Danceability with Best M selected by Cross-Validation')
plt.legend()
plt.grid(True)
plt.show()

# Part 4 - Regularization with lambda = 1/10
lmbda = 1 / 10
regularized_errors = []

for M in M_range:
    phi_train = compute_concatenated_basis_functions(x_train, M)
    phi_test = compute_concatenated_basis_functions(x_test, M)

    # Train the regularized model
    w = np.linalg.lstsq(phi_train.T @ phi_train + lmbda * np.eye(phi_train.shape[1]), phi_train.T @ t_train, rcond=None)[0]

    # Predict on testing set
    y_test = predict(phi_test, w)

    # Calculate mean squared error
    error = calculate_error(y_test, t_test)
    regularized_errors.append(error)

plt.figure(figsize=(10, 6))
plt.plot(M_range, regularized_errors, label='Regularized Testing Error')
plt.xlabel('M')
plt.ylabel('Mean Square Error')
plt.title('Regularized Mean Square Error on Testing Set for Different M')
plt.legend()
plt.grid(True)
plt.show()


phi = basis_function(X_train, M_values)
C = np.linalg.inv(np.dot(phi.T, phi))
beta = np.dot(np.dot(C, phi.T), t_train)

y_hat = np.dot(phi, beta)
print (y_hat)
res = t_train - y_hat
print(res)
RSS = np.sum(res ** 2)
MSE = RSS/len(y_hat)
''' 