import pandas as pd
import numpy as np

# Expected answer. m = 0.05168176, b=18.0465

def gradient_descent(x, y, lr=0.1, epochs=3000):
    # Scale x and y using Min-Max Scaling
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)

    # Initialize parameters
    b = 0.0  # Intercept
    m = 0.0  # Slope
    n = len(y_scaled)  # Number of data points

    # Perform gradient descent
    for epoch in range(epochs):
        y_pred = b + m * x_scaled  # Predicted y values
        error = y_scaled - y_pred  # Error in prediction
        cost = np.mean(error ** 2)   # Mean squared error

        # Calculate gradients
        db = -2 * np.mean(error)  # Derivative w.r.t. intercept b
        dm = -2 * np.mean(error * x_scaled)  # Derivative w.r.t. slope m

        # Update parameters
        b -= lr * db
        m -= lr * dm

        # Optional: Print cost every 100 iterations to monitor progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")

    # Scale back the coefficients to original scale
    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_original = m * (y_max - y_min) / (x_max - x_min)

    return b_original, m_original


if __name__ == "__main__":
    df = pd.read_csv("home_prices.csv")

    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()

    b, m = gradient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")

