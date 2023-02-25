import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac

options = {}
options['t_gamma'] = 0.03
options['t_epsilon'] = 0.03


class run(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        self.add(ResJac(num_nodes=n,options=options),name='ResJac')


if __name__ == '__main__':

    options = {}
    options['wing_area'] = 16.2 # wing area (m^2)

    # run model
    sim = python_csdl_backend.Simulator(run(num_nodes=10))
    sim.run()