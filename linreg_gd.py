import numpy as np
import pandas as pd
import sys
import util
import matplotlib.pyplot as plt

# Check the command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Learning rate
t = 0.001

# Limit interations
max_steps = 1000

# Get the arg and read in the spreadsheet
infilename = sys.argv[1]
X, Y, labels = util.read_excel_data(infilename)
n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")

# Get the mean and standard deviation for each column
## Your code here
mean_allcolumns = X.mean(axis=0)
std_allcolumns = X.std(axis=0)


# Don't mess with the first column (the 1s)
## Your code here
X = X[:, 1:]

# Standardize X to be X'
X_prime = (X - mean_allcolumns[1:]) / std_allcolumns[1:]
X_prime = np.append(np.ones([len(X_prime), 1]), X_prime, axis=1)
# Your code here

# First guess for B is "all coefficents are zero"
# Your code here
B = np.zeros([6])

# Create a numpy array to record avg error for each step
errors = np.array([np.square(np.dot(X_prime, B) - Y).mean()])

for i in range(max_steps):

    # Compute the gradient
    ## Your code here
    gradient = np.dot(X_prime.T, (np.dot(X_prime, B) - Y))

    # Compute a new B (use `t`)
    ## Your code here
    B = B - np.dot(t, gradient)

    # Figure out the average squared error using the new B
    ## Your code here
    avg_square_error = np.square(np.dot(X_prime, B) - Y).mean()

    # Store it in `errors``
    ## Your code here
    errors = np.append(errors, avg_square_error)

    # Check to see if we have converged
    if round(errors[-1], 2) - round(errors[-2], 2) == 0:
        break


print(f"Took {i} iterations to converge")

# "Unstandardize" the coefficients
## Your code here
B_0 = np.subtract(
    B[0], np.dot(B[1:], np.divide(mean_allcolumns[1:], std_allcolumns[1:]))
)
B_1 = B[1:] / std_allcolumns[1:]
np.set_printoptions(formatter={"float_kind": "{:f}".format})
B = np.append(B_0, B_1)
X = np.append(np.ones([len(X), 1]), X, axis=1)

# Show the result
print(util.format_prediction(B, labels))

# Get the R2 score
R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")

# Draw a graph
fig1 = plt.figure(1, (4.5, 4.5))
## Your code here

axs = plt.axes()
axs.plot([i for i in range(errors[:, :1].size)], errors[:, :1])
axs.set_title("Convergence")
axs.set_xlabel("Iterations")
axs.set_ylabel("Mean Squared error")
fig1.savefig("err.png")
