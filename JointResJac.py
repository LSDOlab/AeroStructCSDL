import csdl

class JointResJac(csdl.Model):
    def initialize(self):
        pass
    def define(self):
        # declare variables
        Residuals = self.declare_variable('Residuals')
        beam_list = self.declare_variable('beam_list')
        JointProp = self.declare_variable('JointProp')
        X = self.declare_variable('X')

        # Pre-declaring variables
        n = beam_list['r0'].shape[1]
        Res = SX.sym('Fty', len(JointProp['Child']) * 12, 1)


        for k in range(0, len(JointProp['Child'])):