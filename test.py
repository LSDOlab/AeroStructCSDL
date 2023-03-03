import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac
from inputs import inputs
from BoxBeamRep import *


class test(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('seq')
        self.parameters.declare('bc')
    def define(self):
        n = self.parameters['num_nodes']
        seq = self.parameters['seq']
        bc = self.parameters['bc']

        self.add(inputs(num_nodes=n),name='inputs')

        self.create_input('x',shape=(18,n),val=x)

        self.add(ResJac(num_nodes=n,seq=seq,bc=bc), name='ResJac')


if __name__ == '__main__':
    n = 16
    seq = np.array([3, 1, 2])

    bc = {}
    bc['root'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0])
    bc['tip'] = np.array([8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    sim = python_csdl_backend.Simulator(test(num_nodes=n,seq=seq,bc=bc))
    sim.run()

    Res = sim['Res']
    print(Res)
    print(Res[7,:])