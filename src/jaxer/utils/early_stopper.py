from dataclasses import dataclass

@dataclass
class EarlyStopper:
    max_epochs: int  # max number of epochs without improvement
    n_epochs: int = 0
    optim_value: float = 1e9
    
    def __call__(self, optim_value: float):
        """ Returns True if the training should stop """
        if optim_value < self.optim_value:
            self.optim_value = optim_value
            self.n_epochs = 0
            return False
        
        self.n_epochs += 1 # NOT IMPROVED

        if self.n_epochs >= self.max_epochs:  # STOP TRAINING
            return True
        