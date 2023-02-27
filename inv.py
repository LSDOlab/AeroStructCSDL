import csdl
import numpy as np
import python_csdl_backend


class inv(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        K = self.declare_variable('K',shape=(3,3,n))
        K_inv = self.declare_variable('K_inv',shape=(3,3,n))
        eye = self.declare_variable('eye',shape=(3,3))
        residual = self.create_output('residual',shape=(3,3,n))
        
        for i in range(0,n):
            collapsed_K = csdl.reshape(K[:,:,i],new_shape=(3,3))
            collapsed_K_inv = csdl.reshape(K_inv[:,:,i],new_shape=(3,3))
            residual[:,:,i] = csdl.expand(csdl.matmat(collapsed_K,collapsed_K_inv) - eye, (3,3,1),'ij->ijk')




class solver(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        eye = self.create_input('eye',shape=(3,3),val=np.eye(3))

        solve_inv = self.create_implicit_operation(inv(num_nodes=n))
        solve_inv.declare_state('K_inv', residual='residual')
        solve_inv.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,
        )
        solve_inv.linear_solver = csdl.ScipyKrylov()


        K = self.declare_variable('K',shape=(3,3,n))
        K_inv = solve_inv(K,eye)



if __name__ == '__main__':
    n = 2
    sim = python_csdl_backend.Simulator(solver(num_nodes=n))
    sim['K'] = np.random.rand(3,3,n)
    sim.run()

    print(sim['K_inv'])

    # print partials
    # sim.check_partials(compact_print=True)