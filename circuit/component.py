import numpy as np
from enum import Enum

class component_type(Enum):
    """Enum to define the type of component in the interferometer"""
    BEAM_SPLITTER = 1
    PHASE_SHIFTER = 2
    LOSS_ELEMENT = 3
    BARRIER = 4
    
class component_color(Enum):
    """Enum to define the color of the component in the interferometer"""
    BEAM_SPLITTER = "lime"
    PHASE_SHIFTER = "blueviolet"
    LOSS_ELEMENT = "red"
    BARRIER = "" # Empty since it's not drawn
    LINE = "black"
    OUTPUT_PHASE = "deeppink"
    

class component:
    """This class defines a component in the interferometer.
    It is used to represent a beam splitter, a phase shifter or a loss element.

    Beam splitter:
        - Transformation is:

            e^{iphi}*cos(theta)     -sin(theta)
            e^{iphi}*sin(theta)     cos(theta)
    
        - Modes are specified by modes, which is a tuple of two integers (mode1, mode2).
        - Parameters are specified by params, which is a tuple (theta, phi).


    - Phase shifter is:
        - Transformation is:
        
            e^{iphi}
        
        - Modes are specified by modes, which is a tuple of one integer (mode).
        - Parameters are specified by params, which is a float (phi).
        
        
    - Loss element:
        - Transformation is:
        
            sqrt(loss)
            
        - Modes are specified by modes, which is a tuple of one integer (mode).
        - The loss parameter is not specified when adding the component, but in the construction of the matrix.
        
    - Barrier:
        - No transformation
        - Involved all the modes in the interferometer (we put -1 for the draw function to work).
        - Parameters are not used.
        
    Note: The barrier component is used for alignment purposes and does not affect the transformation matrix.

    All the matrices are applied to the modes specified by modes. and eventually extended to the full matrix size.
    
    Args:
        modes (tuple): the indices of the modes (the first mode is mode 1)
        params (float): the parameters of the component, which depend on the type of component
    """

    def __init__(self,type, modes, params):
        self.type = type
        self.modes = modes
        self.params = params

    def __repr__(self):
        repr = "\n Component type: {} \n Modes: {} and {} \n Parameters: {:.2f}".format(
            self.type,
            self.modes,
            self.params
            )
        return repr