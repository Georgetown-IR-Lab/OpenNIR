from onir import trainers


@trainers.register('trivial')
class TrivialTrainer(trainers.Trainer):
    def path_segment(self):
        return 'trivial'

    def train_batch(self):
        return {
            'losses': {'data': 0},
            'loss_weights': {'data': 0},
        }

    def fast_forward(self, record_count):
        pass
