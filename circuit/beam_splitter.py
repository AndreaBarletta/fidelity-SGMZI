import numpy as np

class beam_splitter:
    """This class defines a beam splitter

    The matrix describing the mode transformation is:

        e^{iphi}*cos(theta)     -sin(theta)
        e^{iphi}*sin(theta)     cos(theta)

    Args:
        mode1 (int): the index of the first mode (the first mode is mode 1)
        mode2 (int): the index of the second mode
        theta (float): the beam splitter angle
        phi (float): the beam splitter phase
    """

    def __init__(self, mode1, mode2, theta, phi):
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta = theta
        self.phi = phi

    def __repr__(self):
        repr = "\n Beam splitter between modes {} and {}: \n Theta angle: {:.2f} \n Phase: {:.2f}".format(
            self.mode1,
            self.mode2,
            self.theta,
            self.phi
            )
        return repr