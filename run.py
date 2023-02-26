import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac


class run(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('options')
    def define(self):
        n = self.parameters['num_nodes']
        options = self.parameters['options']

        self.add(ResJac(num_nodes=n,options=options),name='ResJac')


if __name__ == '__main__':

    options = {}
    options['t_gamma'] = 0.03
    options['t_epsilon'] = 0.03
    

    # run model
    sim = python_csdl_backend.Simulator(run(num_nodes=10,options=options))
    sim.run()