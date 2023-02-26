import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac


class run(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('options')
        self.parameters.declare('seq')
    def define(self):
        n = self.parameters['num_nodes']
        options = self.parameters['options']
        seq = self.parameters['seq']

        self.add(ResJac(num_nodes=n,options=options,seq=seq),name='ResJac')


if __name__ == '__main__':

    options = {}
    options['t_gamma'] = 0.03
    options['t_epsilon'] = 0.03
    seq = np.array([3, 1, 2])
    

    # run model
    sim = python_csdl_backend.Simulator(run(num_nodes=10,options=options,seq=seq))
    sim.run()