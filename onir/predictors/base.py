

# Implementations should either override pred_ctxt or run.
# If predicting requires some state (e.g., pre-loaded data) that is re-used between predictions,
# implement a new pred_ctxt. If not, implement run.

class BasePredictor:
    def pred_ctxt(self):
        return RunPredictorContext(self)

    def run(self, train_ctxt):
        pred_ctxt = self.pred_ctxt()
        if isinstance(pred_ctxt, RunPredictorContext):
            raise NotImplementedError
        return pred_ctxt(train_ctxt)


class RunPredictorContext:
    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, train_ctxt):
        return self.predictor.run(train_ctxt)
