import csdl
import numpy as np

# home of the implicit operation I think

class StickModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']