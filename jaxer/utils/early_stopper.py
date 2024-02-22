from dataclasses import dataclass


@dataclass
class EarlyStopper:
    """Early stopper class

    :param max_epochs: max number of epochs without improvement
    :type max_epochs: int

    :param n_epochs: number of epochs without improvement
    :type n_epochs: int

    :param optim_value: best optimization value
    :type optim_value: float
    """

    max_epochs: int
    n_epochs: int
    optim_value: float
    
    def __call__(self, optim_value: float):
        """ Returns True if the training should stop """
        if optim_value < self.optim_value:
            self.optim_value = optim_value
            self.n_epochs = 0
            return False
        
        self.n_epochs += 1

        if self.n_epochs >= self.max_epochs:
            return True

        return False
        