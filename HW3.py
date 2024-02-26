    # import numpy as np
    # import pandas as pd
    # from math import exp
    # import matplotlib.pyplot as plt
    # from sklearn.model_selection import train_test_split

    # csvloc = "/home/flash/Documents/CS760/titanic_data.csv"
    # data = pd.read_csv(csvloc)
    # df = pd.DataFrame(data)

    # value = [1]*170
    # #inserting a bias 
    # df = df.insert(0,"Atr0",value,False)


    # x = data.iloc[: , :-1]
    # y = data.iloc[: ,  -1]
    # x = np.array(x)
    # y = np.array(y)

    # #Splitting the dataset
    # xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size= 0.15, random_state=0) 
    # # print(data.head())

    # def sigmoid(self, z):

    #         sig_z = (1/(1+np.exp(-z)))
            
    #         assert (z.shape==sig_z.shape), 'Error in sigmoid implementation. Check carefully'
    #         return sig_z

    # def log_likelihood(self, y_true, y_pred):
    #     '''Calculates maximum likelihood estimate
    #     Remember: y * log(yh) + (1-y) * log(1-yh)
    #     Note: Likelihood is defined for multiple classes as well, but for this dataset
    #     we only need to worry about binary/bernoulli likelihood function
    #     Args:
    #         y_true : Numpy array of actual truth values (num_samples,)
    #         y_pred : Numpy array of predicted values (num_samples,)
    #     Returns:
    #         Log-likelihood, scalar value
    #     '''

    #     z = np.dot(X, theta)
    #     # Fix 0/1 values in y_pred so that log is not undefined
    #     y_pred = np.maximum(np.full(y_pred.shape, self.eps), np.minimum(np.full(y_pred.shape, 1-self.eps), y_pred))
    #     likelihood = np.sum(y_true * np.log(1 / (1 + np.exp(-z))) + (1 - y_true) * np.log(1 / (1 + np.exp(z))))
        
    #     return likelihood

    # def fit(self, X, y):
    #         '''Trains logistic regression model using gradient ascent
    #         to gain maximum likelihood on the training data
    #         Args:
    #             X : Numpy array (num_examples, num_features)
    #             y : Numpy array (num_examples, )
    #         Returns: VOID
    #         '''
            
    #         num_examples = X.shape[0]
    #         num_features = X.shape[1]
            
    #         ### START CODE HERE
            
    #         # Initialize weights with appropriate shape
    #         self.weights = np.zeros(num_features)
            
    #         # Perform gradient ascent
    #         for i in range(self.max_iterations):
    #             # Define the linear hypothesis(z) first
    #             # HINT: what is our hypothesis function in linear regression, remember?
    #             z = np.dot(X,self.weights)
    #             # Output probability value by appplying sigmoid on z
    #             y_pred = self.sigmoid(z)
                
    #             # Calculate the gradient values
    #             # This is just vectorized efficient way of implementing gradient. Don't worry, we will discuss it later.
    #             gradient = np.mean((y-y_pred)*X.T, axis=1)
                
    #             # Update the weights
    #             # Caution: It is gradient ASCENT not descent
    #             self.weights = self.weights +self.learning_rate*gradient
                
    #             # Calculating log likelihood
    #             likelihood = self.log_likelihood(y,y_pred)

    #             self.likelihoods.append(likelihood)
        
    #         ### END CODE HERE
                
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, eps=1e-15):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.eps = eps
        self.weights = None
        self.likelihoods = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_likelihood(self, X, y, weights):
        z = np.dot(X, weights)
        y_pred = self.sigmoid(z)
        y_pred_adj = np.clip(y_pred, self.eps, 1 - self.eps)  # Avoid log(0)
        likelihood = np.sum(y * np.log(y_pred_adj) + (1 - y) * np.log(1 - y_pred_adj))
        return likelihood

    def fit(self, X, y):
        num_examples, num_features = X.shape
        self.weights = np.zeros(num_features)
        
        for i in range(self.max_iterations):
            z = np.dot(X, self.weights)
            y_pred = self.sigmoid(z)
            gradient = np.dot(X.T, (y - y_pred)) / num_examples
            self.weights += self.learning_rate * gradient
            likelihood = self.log_likelihood(X, y, self.weights)
            self.likelihoods.append(likelihood)

    def predict(self, X):
        z = np.dot(X, self.weights)
        y_pred = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in y_pred]


from sklearn.model_selection import train_test_split

# Assuming 'data' is your full dataset and 'target' is the column with the labels.

# Split the features and the labels into X and y
X = data.drop('target', axis=1)  # This assumes that 'target' is the name of your column containing the labels
y = data['target']

# Now we split the dataset into training (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# You can now proceed with fitting your model using X_train and y_train
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train, y_train)

# After fitting, you can predict on your test set and evaluate the performance
y_pred = model.predict(X_test)  # You'll need to implement the predict method in your LogisticRegression class

