import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas):
        """
        : u_x: a_x
        : u_y: a_y
        : std_acc: sigma_a
        : x_std_meas:sigma_x
        : y_std_meas: sigma_y
        """

        self.dt = dt

        # Intial State and Control input variables
        self.x = np.matrix([[0], [0], [0], [0]])
        self.u = np.matrix([[u_x],[u_y]])


        # State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    
    def predict(self):

        #x_k =A x_(k-1) + B u_(k-1)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # error covariance P= A*P*A' + Q 
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        kalman_intial_prediction = self.x[0:2].astype('float16').tolist()
        print("Kalman a priori = ",kalman_intial_prediction)

        return self.x[0:2]
    
    def update(self, z):

        
        # Kalman Gain K = P * H'* inv(H*P*H'+R)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P 

        kalman_estimate = self.x[0:2].astype('float16').tolist()
        print("Kalman estimation = ",kalman_estimate)

        return self.x[0:2]