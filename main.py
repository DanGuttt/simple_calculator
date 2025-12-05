import requests
from bs4 import BeautifulSoup
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import sys
sys.setrecursionlimit(10**6)

data = np.loadtxt('train_set_spaced.txt')
X = data[:, :2]
y = data[:, 2].reshape(-1, 1)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

def init(X):
    W=np.random.randn(X.shape[1],1)
    b=np.random.randn(1)
    return(W,b)

def model(X,W,b):
    Z = X.dot(W) + b
    return Z

def mse_loss(A, y):
    return 1/(2*len(y)) * np.sum((A - y)**2)


def gradients(A,X,y):
    dW=(1/len(y))*np.dot(X.T,(A-y))
    db=(1/len(y))*np.sum(A-y)
    return(dW,db)

def update(dW,db,W,b,learning_rate):
    W=W-learning_rate*dW
    b=b-learning_rate*db
    return(W,b)

def predict(X,W,b):
    return model(X,W,b)

def linear_regression(X,y,learning_rate=0.00001,n_iter=100):
    W, b = init(X)

    loss=[]
    for i in range (n_iter):
        A = model(X, W, b)
        loss.append(mse_loss(A,y))
        dW,db=gradients(A,X,y)
        W,b=update(dW,db,W,b,learning_rate)

    y_pred = predict(X, W, b)
    print(f"MSE: {mean_squared_error(y, y_pred):.2f}")
    print(f"R² score: {r2_score(y, y_pred):.3f}")

    return (W,b)


def extraire_nombre_bs4(url):
    response = requests.get(url)
    a = response.text
    a = a[1:-1]
    if response.status_code != 200:
        raise Exception(f"Erreur lors de l'accès au site : {response.status_code}")
    return a


def appel(nb1, nb2):
    url1 = f'https://add-api-71kj.onrender.com/calcm?nombre={nb1}'
    url2 = f'https://add-api-71kj.onrender.com/calcp?nombre={nb2}'
    return int(extraire_nombre_bs4(url1)), int(extraire_nombre_bs4(url2))


def n1Neg(n1, n2):
    return first(n1 + 1, n2) - 1


def n2Neg(n1, n2):
    return first(n1, n2 + 1) - 1


def n1Null(n1, n2):
    return n2


def n2Null(n1, n2):
    return n1


def first(n1, n2):
    if n1 < 0:
        return n1Neg(n1, n2)
    if n2 < 0:
        return n2Neg(n1, n2)
    if n1 == 0:
        return n1Null(n1, n2)
    if n2 == 0:
        return n2Null(n1, n2)

    return first(n1 - 1, n2) + 1


def fonction(n1, n2):
    print("Nous allons faire l'addition")
    return first(n1, n2)


def calculatrice_quantique(a, b):
    a_cubed = a ** 3
    b_cubed = b ** 3

    if a_cubed >= 1:
        base = 2 ** round(math.log2(a_cubed))
        epsilon_ratio = (a_cubed - base) / base
    else:
        base = 1
        epsilon_ratio = a_cubed - 1

    base_cbrt = base ** (1 / 3)
    x = epsilon_ratio
    series_a = 1.0

    for k in range(1, 12):
        coef = 1.0
        for j in range(k):
            coef *= (1 / 3 - j) / (j + 1)
        term = coef * (x ** k)
        series_a += term

    a_approx = base_cbrt * series_a

    x_newton = b * 1.1
    for i in range(6):
        f_x = x_newton ** 3 - b_cubed
        f_prime_x = 3 * (x_newton ** 2)
        x_new = x_newton - f_x / f_prime_x
        x_newton = x_new

    b_approx = x_newton

    exp_a = math.floor(math.log10(abs(a_approx))) if a_approx != 0 else 0
    mantissa_a = a_approx / (10 ** exp_a) if a_approx != 0 else 0

    exp_b = math.floor(math.log10(abs(b_approx))) if b_approx != 0 else 0
    mantissa_b = b_approx / (10 ** exp_b) if b_approx != 0 else 0

    if exp_a == exp_b:
        mantissa_sum = mantissa_a + mantissa_b
        result_intermediate = mantissa_sum * (10 ** exp_a)
    else:
        if exp_a > exp_b:
            b_adjusted = b_approx
            result_intermediate = a_approx + b_adjusted
        else:
            a_adjusted = a_approx
            result_intermediate = a_adjusted + b_approx

    epsilon_correction = 0.000001 * (result_intermediate % 0.01)
    correction = 1 + epsilon_correction - (epsilon_correction ** 2) / 2
    final_result = result_intermediate * correction

    power7 = final_result ** 7
    root7 = power7 ** (1 / 7)

    # return round(final_result)
    print(final_result)
    return(round(final_result))


W,b=linear_regression(X,y,learning_rate=0.1,n_iter=10000)

n1 = float(input("premier nb: "))
n2 = float(input("deuxième nb: "))

def addition(X,Y,X_mean,X_std, W,b,y_std,y_mean):
    new_data = np.array([[X, Y]])
    new_data_normalized = (new_data - X_mean) / X_std
    prediction_normalized = predict(new_data_normalized, W, b)
    prediction = prediction_normalized * y_std + y_mean
    print(f"\nPrédiction pour {new_data[0]}: {prediction[0][0]:.2f}")

addition((calculatrice_quantique(fonction(appel(n1, n2)[0], appel(n1, n2)[1]) - 5, 5))-5,5,X_mean,X_std, W,b,y_std,y_mean)
