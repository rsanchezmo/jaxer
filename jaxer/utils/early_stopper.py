

class EarlyStopper:
    """Early stopper class. Stops the training if the optimization metric does not improve for a number of epochs

    :param max_epochs: max number of epochs without improvement
    :type max_epochs: int
    """

    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs
        self.n_epochs = 0
        self.optim_value = 1.e9
    
    def __call__(self, optim_value: float):
        """ Returns True if the training should stop

        :param optim_value: value of the optimization metric
        :type optim_value: float

        :returns: True if the training should stop
        :rtype: bool
        """
        if optim_value < self.optim_value:
            self.optim_value = optim_value
            self.n_epochs = 0
            return False
        
        self.n_epochs += 1

        if self.n_epochs >= self.max_epochs:
            return True

        return False
        