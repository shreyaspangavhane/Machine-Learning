import numpy as np;
import pandas as pd;

def gredient_descent(x,y,lr=0.01,epochs=3000):
    m,b=0.0,0.0
    # scale x and  y using min max scaling
    x_min,x_max=x.min(),x.max()
    y_min,y_max=y.min(),y.max()

    x_scale=(x-x_min)/(x_max-x_min)
    y_scale=(y-y_min)/(y_max-y_min)


    for epoch in range(epochs):
        y_predict=m*x_scale+b

        error= y_scale-y_predict         ## error= (y-cap(y))^2
        cost= np.mean(error**2)   

        dm= -2*(np.mean(error*x))
        db= -2*(np.mean(error))

        b = b - db *lr
        m = m - dm *lr

        if epoch%100 ==0:
            print(f"m = {m}, b= {b}, Epoch= {epoch},Cost={cost} ")
        
        # scale back the coefficients to original scale
        b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
        m_original = m * (y_max - y_min) / (x_max - x_min)

        return b_original, m_original


if __name__=="__main__":

    # x=np.array([1,2,3,4,5])
    # y=np.array([5,7,9,11,13])
    df=pd.read_csv("./home_prices.csv")
    # print(df)

    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()
    b, m = gredient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")

