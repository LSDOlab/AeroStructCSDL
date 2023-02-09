import csdl
import numpy as np


class aero(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        X = self.declare_variable('X')
        Xd = self.declare_variable('Xd')
        F = self.declare_variable('F') # forces
        M = self.declare_variable('M') # moments