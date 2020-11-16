import numpy as np
import cv2


def get_link_matrix(theta, d, a, alpha):
    Rz = np.matrix([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    TransZ = np.matrix([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.0, d], [0, 0, 0, 1.0]])
    TransX = np.matrix([[1.0, 0, 0, a], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    Rx = np.matrix([[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])
    link_matrix = Rz.dot(TransZ).dot(TransX).dot(Rx)
    return(link_matrix)

def get_transormation_matrix(ja1, ja2, ja3, ja4):
    TC = get_link_matrix(np.pi/2, 0, 0, 0)
    T1 = get_link_matrix(ja1, 2.5, 0, np.pi/2)
    T2 = get_link_matrix(np.pi/2, 0, 0, 0)
    T3 = get_link_matrix(ja2, 0, 0, np.pi/2)
    T4 = get_link_matrix(ja3, 0, 3.5, -np.pi/2)
    T5 = get_link_matrix(ja4, 0, 3, 0)
    T0 = TC.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5)
    T0 = np.round(T0, 2)

    return T0

ja1 = 1
ja2 = 0
ja3 = 0
ja4 = 0

x_d = np.array([0,0,0,0,0])

x_d

possible_ja1 = np.linspace(-np.pi, np.pi, 100)
possible_ja2 = np.linspace(-np.pi/2, np.pi/2, 50)

for ja1 in possible_ja1:
    for ja2 in possible_ja2:
        T0 = get_transormation_matrix(ja1, ja2, ja3, ja4)
        x_e = np.array([T0[:3,3]])
        angles = np.array([ja1, ja2])
        x_e = np.append(angles, x_e)
        x_d = np.concatenate((x_e, x_d), axis=0)

np.savetxt("foo.csv", x_d, delimiter=",")
