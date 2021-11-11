import numpy as np


class IntegrableNumber:
    fps   = 30  # Frames per second
    order = 2   # number of derivatives for x
    coef  = 0.1 # coefficient used to update -> x = prev_x + dframe * coef * new_x
    
    def __init__(self):
        self.x          = np.zeros(1 + IntegrableNumber.order)  # array of [x, dx, d²x, ...]
        self.computed   = np.zeros(1 + IntegrableNumber.order) # array to say if the derivatives have been computed
        self.last_frame = -1                                    # index of the last frame used to update
    
    
    def update(self, value, frame_index):
        """
        Updates the derivatives
        
        Parameters
        ----------
        value      : (float) new value
        frame_index : (int)  index of the frame associated to the value

        Returns
        -------
        None.
        """
        
        # Frames between the previous update and this one
        dframe = float(frame_index - self.last_frame)
        
        # New values
        new_x    = np.zeros(1 + IntegrableNumber.order)
        new_x[0] = value
        new_computed    = np.zeros(1 + IntegrableNumber.order)
        new_computed[0] = 1
        
        for i in range(IntegrableNumber.order):
            new_derivative = (new_x[i] - self.x[i]) / float(dframe)
            if self.computed[i]:
                new_x[i+1]  = (1. - IntegrableNumber.coef) * self.x[i] + dframe*IntegrableNumber.coef*self.computed[i] * new_derivative
                new_x[i+1] /= (1. - IntegrableNumber.coef) *    1.     + dframe*IntegrableNumber.coef*self.computed[i]
            else:
                new_x[i+1] = self.x[i+1]
            if self.computed[i]: new_computed[i+1] = 1
            
        # Update
        self.x = new_x
        self.computed = new_computed
        self.last_frame = frame_index
        
    
    def predict(self, frame_index):
        """
        Return the expected value of x using its derivatives at the frame index given.
        
        Parameters
        ----------
        frame_index : (int)  index of the frame associated to the value

        Returns
        -------
        expected value
        """
        
        # Frames between the previous update and this one
        dframe = float(frame_index - self.last_frame)
        
        # We integrate one after the other
        v = self.x[-1]
        for i in range(IntegrableNumber.order):
            index = IntegrableNumber.order - i - 1
            v = self.x[index] + v*dframe
            
        return v