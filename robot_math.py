import numpy as np
from sympy import *
############ if not symbolic，uncomment the following
# import math
# def cos(x):
#     return math.cos(x)

# def sin(x):
#     return math.sin(x)
##############
def rotX4(x):
    T=np.array([[1 ,  0 ,    0,  0],
                [0 , cos(x), -sin(x), 0],
                [0 , sin(x) , cos(x) ,0],
                [0 , 0,   0, 1]])
    return T

def rotX(x):
    T=np.array([[1 ,  0 ,    0],
                [0 , cos(x), -sin(x)],
                [0 , sin(x) , cos(x)]])
    return T

def rotY4(th):
    rot=np.array([[cos(th) ,  0,     sin(th) , 0],
                  [0  ,       1,     0 ,       0],
                  [-sin(th)  ,0,     cos(th),  0],
                  [0, 0, 0 ,1]])
    return rot

def rotY(th):
    rot=np.array([[cos(th) ,  0,     sin(th) ],
                  [0  ,       1,     0 ,     ],
                  [-sin(th)  ,0,     cos(th) ]])
    return rot

def rotZ4(th):
    rot=np.array([[cos(th), -sin(th), 0, 0],
                  [sin(th) , cos(th), 0, 0],
                  [0 ,       0  ,     1 ,0],
                  [0, 0, 0, 1]])
    return rot

def rotZ(th):
    rot=np.array([[cos(th), -sin(th), 0],
                  [sin(th) , cos(th), 0],
                  [0 ,       0  ,     1]])
    return rot
