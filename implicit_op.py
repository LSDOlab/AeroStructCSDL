import csdl
import numpy as np
import python_csdl_backend
from ResJac import ResJac


class SolveNonlinear(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']


        # define the internal model that defines a residual
        model = csdl.Model()
        model.add(ResJac(num_nodes=n,),name='ResJac')


        solve_nonlinear = self.create_implicit_operation(model)
        solve_nonlinear.declare_state('x', residual='RES')
        solve_nonlinear.nonlinear_solver = csdl.NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_nonlinear.linear_solver = csdl.ScipyKrylov()