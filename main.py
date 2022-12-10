import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


class LinearRegression:
    # h(x) = X*w+p , w => weights(theta0,1,2,3,4,...)
    def __init__(self, x_train, y_train, w, numOfTrainingItems, learning_rate):
        self.x_train = x_train
        self.y_train = y_train
        self.w = w
        self.numOfTrainingItems = numOfTrainingItems
        self.learning_rate = learning_rate

    # Hypothesis Function
    def hypothesis(self, w, x):
        return np.dot(x, w)

    # Cost Function
    def Cost_Function(self):
        hypotheses = self.hypothesis(self.w, self.x_train)
        errors = hypotheses - self.y_train
        total_error = np.sum(errors * errors)
        return total_error / (2 * float(self.numOfTrainingItems))

    # get the values of theta by gradient descent
    def gradient_descent(self):
        hypothesis = self.hypothesis(self.w, self.x_train)
        errors = hypothesis - self.y_train
        self.w -= (self.learning_rate / self.numOfTrainingItems) * self.x_train.transpose().dot(errors)
        return self.w

    # training the data
    def train(self):
        costs = np.zeros(iterations)
        for i in range(iterations):
            self.w = self.gradient_descent()
            costs[i] = self.Cost_Function()
        return self.w, costs

    # predict new values
    def predict(self, x_test):
        hypothesis = self.hypothesis(self.w, x_test)
        return hypothesis

    def calc_accuracy(self, y_test, y_predicted):
        return 1 - (sum((y_test - y_predicted) ** 2) / sum((y_test - np.mean(y_test)) ** 2))


# loading the data (req1)
data = pd.read_csv("car_data.csv")
data.head()

# # plotting to chose best features
# for column in data:
#     plot.figure(figsize=(3,3))
#     plot.scatter(data[column],data['price'])
#     plot.xlabel('price')
#     plot.ylabel(column)
#     plot.show()


# choosing best features affect on data
new_data = data[['curbweight', 'enginesize', 'citympg', 'highwaympg', 'price']]
# 2 positive and 2 negative


# shuffle data
new_data = new_data.sample(frac=1)

y_data = new_data.values[:, -1]
x_data = new_data.iloc[:, 0:4]

# normalize data
x_data = x_data.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
x_data.insert(0, 'x0', 1)

x = x_data.to_numpy().reshape((-1, 5))
minVal = min(y_data)
maxVal = max(y_data)

for i in range(len(y_data)):
    y_data[i] = (y_data[i] - minVal) / (maxVal - minVal)


# Split the dataset into training and testing sets (req2)
x_train = x[0:160]
y_train = y_data[0:160]
x_test = x[160:]
y_test = y_data[160:]

m = x_train.shape[0]  # num of rows

# We need theta parameter for every input variable.
theta = [np.random.random(), np.random.random(), np.random.random(), np.random.random(), np.random.random()]
iterations = 1500
alpha = 0.005

# run linear regression
linearReg = LinearRegression(x_train, y_train, theta, m, alpha)
thetas, cost_history = linearReg.train()
print('Final values of theta =', thetas)
# print(cost_history)


# predictions on new data (req4)
Y_Predict = linearReg.predict(x_test)
# print(f"{Y_Predict}")

# Calculate the accuracy (req5)
print(f"Accuracy : {round(linearReg.calc_accuracy(y_test, Y_Predict) * 100, 1)}%")

# plotting the output data
plot.plot(range(1, iterations + 1), cost_history, color='red')
plot.rcParams["figure.figsize"] = (10, 6)
plot.grid()
plot.xlabel("Number of iterations")
plot.ylabel("cost (J)")
plot.title("gradient descent")
plot.show()
