import csdl
import python_csdl_backend
import numpy as np


class CalcNodalK(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        K = self.create_output('K',shape=(3,3,n))
        Ka = self.create_output('Ka',shape=(3,3,n-1))

        for i in range(0, n):
            # surface beam
            if not (seq - [1, 3, 2]).any():
                K[i][0, 0] = cos(th[2, i]) * cos(th[1, i])
                K[i][0, 1] = 0
                K[i][0, 2] = -sin(th[1, i])
                K[i][1, 0] = -sin(th[2, i])
                K[i][1, 1] = 1
                K[i][1, 2] = 0
                K[i][2, 0] = cos(th[2, i]) * sin(th[1, i])
                K[i][2, 1] = 0
                K[i][2, 2] = cos(th[1, i])
            # fuselage beam
            if not (seq - [3, 1, 2]).any():
                K[i][0, 0] = cos(th[1, i])
                K[i][0, 1] = 0
                K[i][0, 2] = -cos(th[0, i]) * sin(th[1, i])
                K[i][1, 0] = 0
                K[i][1, 1] = 1
                K[i][1, 2] = sin(th[0, i])
                K[i][2, 0] = sin(th[1, i])
                K[i][2, 1] = 0
                K[i][2, 2] = cos(th[0, i]) * cos(th[1, i])
            if i >= 1:
                Ka[i - 1][:, :] = (K[i - 1][:, :] + K[i][:, :]) / 2