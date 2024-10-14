class EarlyStopper:
    def __init__(self, patience=200, counter_init=0, acc_init=0., min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = counter_init
        self.max_val_acc = acc_init
    
    def early_stop(self, val_acc):
        """ For early stopping criteria. 
        If current val_acc < best_acc and counter == patience, then return true and stop the training.
        Otherwise, reset the counter to 0, replace the best_acc and return false, countinue training.
        """
        if val_acc > self.max_val_acc:
            self.max_val_acc = val_acc
            self.counter = 0
        elif val_acc < (self.max_val_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False