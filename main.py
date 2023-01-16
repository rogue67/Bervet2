import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
'''
 This excersice tests Euler forvard anf Heun's methon 
 for solving ODE.
'''

def rhsode(t, y):
    return y - 0.5*np.exp(t/2)*np.sin(5*t) + 5*np.exp(t/2)*np.cos(5*t)

def solverDemo():
    y0 = 0
    t0 = 0
    t1 = np.pi
    h = float(input("Give me h: "))
    n = int((t1 - t0)/h)
    T = np.linspace(t0, t1, n)
    Y_euler = [y0]
    Y_heun = [y0]

    for t in T[1:]:
        y = y0 + h*rhsode(t, y0)
        Y_euler.append(y)
        y0 = y

    y0 = 0
    for t in T[1:]:
        k1 = rhsode(t, y0)
        k2 = rhsode(t + h, y0 + h*k1)
        y = y0 + h*(k1 + k2)/2
        Y_heun.append(y)
        y0 = y

    y0 = 0
    t_span = [t0, t1]
    sol = integrate.solve_ivp(fun=rhsode, t_span=[t0, t1], y0=[y0], atol=1e-8, rtol=1e-8)
    plt.plot(T, Y_euler, 'r')
    plt.plot(T, Y_heun, 'b')
    plt.plot(sol.t, sol.y[0], '.')
    plt.xlabel('t');
    plt.ylabel('y(t)')
    plt.show()


if __name__ == '__main__':
    solverDemo()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
