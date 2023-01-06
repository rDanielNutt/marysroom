import numpy as np

class Loss:
    def __init__(self, ap=np):
        self.ap = ap

    def calculate(self, preds, true):
        sample_losses = self.forward(self.ap.array(preds), self.ap.array(true))
        batch_loss = self.ap.mean(sample_losses)
        if self.ap == np:
            return batch_loss
        else:
            return batch_loss.get()


class CATCROSSENTROPY(Loss):
    def forward(self, preds, true):
        pred_clipped = self.ap.clip(preds, 1e-8, 1-1e-8)
        neg_log_probs = - ((self.ap.log(pred_clipped) * true) + 
                           (self.ap.log(1 - pred_clipped) * (1 - true)))

        return self.ap.sum(neg_log_probs, axis=1)


    def backprop(self, preds, true):
        pred_clipped = self.ap.clip(preds, 1e-8, 1-1e-8)
        return -((pred_clipped - true)/(pred_clipped**2 - pred_clipped))


class MSE(Loss):
    def forward(self, pred, true):
        return self.ap.sum((pred - true)**2, axis=1)

    def backprop(self, pred, true):
        return (2*(pred - true))


