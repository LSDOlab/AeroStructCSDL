import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac
from inputs import inputs


class implicit_op(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('seq')
        self.parameters.declare('bc')
    def define(self):
        n = self.parameters['num_nodes']
        seq = self.parameters['seq']
        bc = self.parameters['bc']

        self.add(inputs(num_nodes=n),name='inputs')

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
    seq = np.array([3, 1, 2])

    bc = {}
    bc['root'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.5707963267098655, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0])
    bc['tip'] = np.array([8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 8888.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])

    sim = python_csdl_backend.Simulator(implicit_op(num_nodes=n,seq=seq,bc=bc))
    sim.run()

    print(sim['x'])