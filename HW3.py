import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


data_path = 'titanic_data.csv'
titanic_data = pd.read_csv(data_path)


X = titanic_data.drop('Survived', axis=1).values
y = titanic_data['Survived'].values


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def normalize_features(X):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X[:, 1:])
    return np.concatenate((np.ones((X_normalized.shape[0], 1)), X_normalized), axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = add_intercept(X_train)
X_test = add_intercept(X_test)
X_train = normalize_features(X_train)
X_test = normalize_features(X_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    z = np.dot(X, theta)
    return np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))

def gradient_of_log_likelihood(X, y, theta):
    z = np.dot(X, theta)
    return np.dot(X.T, y - sigmoid(z))

def gradient_ascent(X, y, learning_rate, num_iterations):
    start_time = time.time()
    theta = np.zeros(X.shape[1])
    for step in range(num_iterations):
        gradient = gradient_of_log_likelihood(X, y, theta)
        theta += learning_rate * gradient
        if np.allclose(gradient, np.zeros_like(gradient)):
            break  # Convergence condition: gradient is close to zero
    else:
        step = num_iterations  # If loop completes without break, set step to maximum iterations
    end_time = time.time()
    convergence_time = end_time - start_time
    return theta, convergence_time, step


learning_rate = 0.001
num_iterations = 10000


theta_hat, convergence_time, convergence_steps = gradient_ascent(X_train, y_train, learning_rate, num_iterations)





def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return probabilities >= 0.5


def calculate_accuracy(X, y, theta):
    predictions = predict(X, theta)
    return (predictions == y).mean()


test_predictions = predict(X_test, theta_hat)



comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_predictions.astype(int)
})


comparison_df.to_csv('test_predictions.csv', index=False)



train_accuracy = calculate_accuracy(X_train, y_train, theta_hat)
test_accuracy = calculate_accuracy(X_test, y_test, theta_hat)


log_likelihood_value = log_likelihood(X_train, y_train, theta_hat)
print("Log-likelihood of optimized theta:", log_likelihood_value)

print("Convergence time: {:.2f} seconds".format(convergence_time))
print("Convergence steps:", convergence_steps)

print("Optimized theta:", theta_hat)

print ("train and test accuracies :", train_accuracy,test_accuracy)


feature_vector = np.array([[3, 0, 24, 0, 0, 9]])
feature_vector_normalized = normalize_features(feature_vector)
feature_vector_normalized_with_intercept = np.hstack([np.ones((feature_vector_normalized.shape[0], 1)), feature_vector_normalized])

print (predict(feature_vector_normalized_with_intercept, theta_hat))
