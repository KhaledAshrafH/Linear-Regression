# Linear Regression from Scratch in Python

This program implements linear regression from scratch using the gradient descent algorithm in Python. It predicts car prices based on selected features and uses a dataset of cars with their respective prices.

## Dataset

The program uses the "car_data.csv" dataset, which contains 205 records of cars with 25 features per record in addition to 1 target column. These features include the car size and dimensions, the fuel system and fuel type used by the car, the engine size and type, the horsepower, etc. The final column (i.e., the target) is the car price

## Features Selection

The program selects four numerical features that are positively or negatively correlated to the car price. These features are:

- Curb weight
- Engine size
- City MPG
- Highway MPG
- Price

`2 positive and 2 negative`

## Data Preprocessing

Before applying linear regression, the program performs some data preprocessing steps, including:

- Normalizing the feature data using min-max scaling
- Shuffling the dataset's rows
- Splitting the dataset into training and testing sets

## Linear Regression

The program implements linear regression from scratch using gradient descent to optimize the parameters of the hypothesis function. The hypothesis function is:

h(x) = X * w + p

where X is the feature matrix, w is the weight vector, and p is the bias term.

The program initializes the weight vector w and the bias term p randomly and updates them iteratively using gradient descent until convergence or a maximum number of iterations is reached.

The program also calculates the cost (mean squared error) in every iteration to visualize the learning curve and check the convergence of the gradient descent algorithm.

## How to Run

To run this program, follow these steps:

1. Install Python 3 on your system.
2. Install the required libraries: numpy, pandas, matplotlib, and sklearn.
3. Place the "car_data.csv" file in your working directory.
4. Open a terminal or command prompt.
5. Navigate to the directory containing the program files.
6. Execute the following command:
   
   ```
   python main.py
   ```
8. The program will display various outputs, including scatter plots, cost function plots, final parameter values, predictions, and the R^2 score for the testing set.

## Customization

You can customize the program by modifying the following parameters:

- Number of iterations for gradient descent
- Learning rate values to try
- Features used for linear regression
- Ratio of training and testing sets

To make changes, edit the corresponding variables in the code.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


## Team

- [Khaled Ashraf Hanafy Mahmoud - 20190186](https://github.com/KhaledAshrafH).
- [Noura Ashraf Abdelnaby Mansour - 20190592](https://github.com/NouraAshraff).

## License

This program is licensed under the [MIT License](LICENSE.md).
