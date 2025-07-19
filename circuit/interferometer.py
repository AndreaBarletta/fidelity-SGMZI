import numpy as np
from .component import component as cmp
from .component import component_type as cmp_type
from .component import component_color as cmp_color

class interferometer:
    """This class defines an interferometer. 

    An interferometer contains an ordered list of components, represented here by comp_list.
    For comp in comp_list, comp.modes correspond to the labels of the modes (which start at 1). 
    The interferometer also contains a list of output phases described by output_phases.
    """

    def __init__(self):
        self.comp_list = []
        self.output_phases = []

    def count_modes(self):          
        """Calculate number of modes involved in the transformation. 

        This is required for calculate_transformation and draw

        Returns:
            the number of modes in the transformation
        """
        highest_index = max([max(comp.modes) for comp in self.comp_list])
        return highest_index

    def calculate_transformation(self, LN_dB=0.0):
        """Calculate unitary matrix describing the transformation implemented by the interferometer
    
        Returns:
            complex-valued 2D numpy array representing the interferometer
        """
        N = int(self.count_modes())
        U = np.eye(N, dtype=np.complex128)

        for comp in self.comp_list:
            T = np.eye(N, dtype=np.complex128)
            if comp.type == cmp_type.BEAM_SPLITTER:
                T[comp.modes[0] - 1, comp.modes[0] - 1] = np.exp(1j * comp.params[1]) * np.cos(comp.params[0])
                T[comp.modes[0] - 1, comp.modes[1] - 1] = -np.sin(comp.params[0])
                T[comp.modes[1] - 1, comp.modes[0] - 1] = np.exp(1j * comp.params[1]) * np.sin(comp.params[0])
                T[comp.modes[1] - 1, comp.modes[1] - 1] = np.cos(comp.params[0])
            elif comp.type == cmp_type.PHASE_SHIFTER:
                T[comp.modes[0] - 1, comp.modes[0] - 1] = np.exp(1j * comp.params[0])
            elif comp.type == cmp_type.LOSS_ELEMENT:
                loss = 10 ** (-LN_dB / 10)  # Convert dB to linear scale
                T[comp.modes[0] - 1, comp.modes[0] - 1] = np.sqrt(loss)
            elif comp.type == cmp_type.BARRIER:
                # Barrier does not change the transformation matrix, so we skip it
                continue
            else:
                raise ValueError("Unknown component type: {}".format(comp.type))
                
            U = np.matmul(T,U)

        while np.size(self.output_phases) < N:  # Autofill for users who don't want to bother with output phases
            self.output_phases.append(0)

        D = np.diag(np.exp([1j * phase for phase in self.output_phases]))
        U = np.matmul(D,U)
        return U

    def draw(self, show_plot=True):  
        """Use matplotlib to make a drawing of the interferometer

        Args:
            show_plot (bool): whether to show the generated plot
        """

        import matplotlib.pyplot as plt
        plt.figure()
        N = self.count_modes()
        mode_tracker = np.zeros(N)

        # Draw the input lines
        for ii in range(N):
            plt.plot((-1, 0), (ii, ii), lw=1, color=cmp_color.LINE.value)

        # Draw the beam splitters / phase shifters
        for comp in self.comp_list:
            if comp.type == cmp_type.BEAM_SPLITTER:
                color = cmp_color.BEAM_SPLITTER.value
                x = np.max([mode_tracker[comp.modes[0] - 1], mode_tracker[comp.modes[1] - 1]])
                # Draw the beam splitter lines
                plt.plot((x+0.3, x+1), (N - comp.modes[0], N - comp.modes[1]), lw=1, color=color)
                plt.plot((x, x+0.3), (N - comp.modes[1], N - comp.modes[1]), lw=1, color=color)
                plt.plot((x+0.3, x+1), (N - comp.modes[1], N - comp.modes[0]), lw=1, color=color)
                plt.plot((x+0.4, x+0.9), (N - (comp.modes[1] + comp.modes[0])/2, N - (comp.modes[1] + comp.modes[0])/2), lw=1, color=color)
                reflectivity = "{:2f}".format(np.cos(comp.params[0])**2)
                plt.text(x+0.9, N + 0.05 - (comp.modes[1] + comp.modes[0])/2, reflectivity[0:3], color=color, fontsize=7)
                # If the input phase shifter is present (i.e. phi!=0), draw it
                if comp.params[1] != 0:
                    plt.plot((x, x+0.3), (N - comp.modes[0], N - comp.modes[0]), lw=1, color=color)
                    plt.plot((x+0.15, x+0.15), (N+0.3-(comp.modes[1] + comp.modes[0])/2., N+0.7-(comp.modes[1] + comp.modes[0])/2.), lw=1, color=color)
                    circle = plt.Circle((x+0.15, N+0.5-(comp.modes[1] + comp.modes[0])/2.), 0.1, fill=False, color=color)
                    plt.gca().add_patch(circle)
                    # Draw the phase label
                    phase = "{:2f}".format(comp.params[1])
                    if comp.params[1] > 0:
                        plt.text(x+0.2, N+0.7-(comp.modes[1] + comp.modes[0])/2., phase[0:3], color=color, fontsize=7)
                    else:
                        plt.text(x+0.2, N+0.7-(comp.modes[1] + comp.modes[0])/2., phase[0:4], color=color, fontsize=7)
                else:
                    plt.plot((x, x+0.3), (N - comp.modes[0], N - comp.modes[0]), lw=1, color=cmp_color.LINE.value)
                # Draw filler lines
                if x > mode_tracker[comp.modes[0]-1]:
                    plt.plot((mode_tracker[comp.modes[0]-1], x), (N-comp.modes[0], N-comp.modes[0]), lw=1, color=cmp_color.LINE.value)
                if x > mode_tracker[comp.modes[1]-1]:
                    plt.plot((mode_tracker[comp.modes[1]-1], x), (N-comp.modes[1], N-comp.modes[1]), lw=1, color=cmp_color.LINE.value)
                #Update the mode tracker
                mode_tracker[comp.modes[0] - 1] = x + 1
                mode_tracker[comp.modes[1] - 1] = x + 1
                
            elif comp.type == cmp_type.PHASE_SHIFTER:
                color = cmp_color.PHASE_SHIFTER.value
                x = mode_tracker[comp.modes[0] - 1]
                # Draw the phase shifter (for understanding purposes, we will draw it even if phi = 0)
                plt.plot((x, x+0.3), (N - comp.modes[0], N - comp.modes[0]), lw=1, color=color)
                plt.plot((x+0.15, x+0.15), (N-0.2-comp.modes[0], N+0.2-comp.modes[0]), lw=1, color=color)
                circle = plt.Circle((x+0.15, N-comp.modes[0]), 0.1, fill=False, color=color)
                plt.gca().add_patch(circle)
                # Draw the phase label
                phase = "{:2f}".format(comp.params[0])
                if comp.params[0] > 0:
                    plt.text(x+0.2, N+0.2-comp.modes[0], phase[0:3], color="blueviolet", fontsize=7)
                else:
                    plt.text(x+0.2, N+0.2-comp.modes[0], phase[0:4], color="blueviolet", fontsize=7)
                #Update the mode tracker
                mode_tracker[comp.modes[0] - 1] = x + 0.3
                
            elif comp.type == cmp_type.LOSS_ELEMENT:
                color = cmp_color.LOSS_ELEMENT.value
                x = mode_tracker[comp.modes[0] - 1]
                # Draw the phase shifter (for understanding purposes, we will draw it even if phi = 0)
                plt.plot((x, x+0.3), (N - comp.modes[0], N - comp.modes[0]), lw=1, color=color)
                plt.plot((x+0.15, x+0.15), (N-0.2-comp.modes[0], N+0.2-comp.modes[0]), lw=1, color=color)
                circle = plt.Circle((x+0.15, N-comp.modes[0]), 0.1, fill=False, color=color)
                plt.gca().add_patch(circle)
                #Update the mode tracker
                mode_tracker[comp.modes[0] - 1] = x + 0.3
            elif comp.type == cmp_type.BARRIER:
                # Update the mode tracker
                mode_tracker[comp.modes[0] - 1] = x + 0.3
            else:
                raise ValueError("Unknown component type: {}".format(comp.type))
            

        # Draw the output phase shifters
        max_x = np.max(mode_tracker)
        color = cmp_color.OUTPUT_PHASE.value
        for ii in range(N):
            plt.plot((mode_tracker[ii], max_x+1), (N-ii-1, N-ii-1), lw=1, color=cmp_color.LINE.value)
            while np.size(self.output_phases) < N:  # Autofill for users who don't want to bother with output phases
                self.output_phases.append(0)
            if self.output_phases[ii] != 0:
                plt.plot((max_x+0.5, max_x+0.5), (N-ii-1.2, N-ii-0.8), lw=1, color=color)
                circle = plt.Circle((max_x+0.5, N-ii-1), 0.1, fill=False, color=color)
                plt.gca().add_patch(circle)
                phase = str(self.output_phases[ii])
                if self.output_phases[ii] > 0:
                    plt.text(max_x+0.6, N-ii-0.8, phase[0:3], color=color, fontsize=7)
                else:
                    plt.text(max_x+0.6, N-ii-0.8, phase[0:4], color=color, fontsize=7)

        plt.text(-1, N-0.3, "Light in", fontsize=10)
        plt.text(max_x+0.5, N-0.3, "Light out", fontsize=10)
        plt.gca().axes.set_ylim([-1.8, N+0.2])
        plt.axis("off")
        if show_plot:
            plt.show()