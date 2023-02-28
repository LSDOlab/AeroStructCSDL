import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac


class implicit_op(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('options')
        self.parameters.declare('seq')
        self.parameters.declare('bc')
    def define(self):
        n = self.parameters['num_nodes']
        options = self.parameters['options']
        seq = self.parameters['seq']
        bc = self.parameters['bc']

        eye = self.create_input('eye',shape=(3,3),val=np.eye(3))
        t_epsilon = self.create_input('t_epsilon',shape=(1,1),val=options['t_epsilon'])
        t_gamma = self.create_input('t_gamma',shape=(1,1),val=options['t_gamma'])

        solve_res = self.create_implicit_operation(ResJac(num_nodes=n,seq=seq,bc=bc))
        solve_res.declare_state('x', residual='Res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()

        ans = solve_res()




if __name__ == '__main__':
    n = 16

    options = {}
    options['t_gamma'] = 0.03
    options['t_epsilon'] = 0.03
    seq = np.array([3, 1, 2])

    bc = {}
    bc['root'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.5707963267098655, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0])
    bc['tip'] = np.array([8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])

    sim = python_csdl_backend.Simulator(implicit_op(num_nodes=n,options=options,seq=seq,bc=bc))
    sim.run()

    print(sim['x'])
    print(sim['Res'])