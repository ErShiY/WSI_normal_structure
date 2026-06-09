class EarlyStopping:
    def __init__(self, monitor="macro_f1", mode="max", patience=8, min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.should_stop = False

        if mode == "max":
            self.best_score = -float("inf")
        elif mode == "min":
            self.best_score = float("inf")
        else:
            raise ValueError(f"Unsupported early stopping mode: {mode}")

    def is_improved(self, current_score):
        if self.mode == "max":
            return current_score > self.best_score + self.min_delta
        else:
            return current_score < self.best_score - self.min_delta

    def step(self, metrics):
        current_score = metrics[self.monitor]

        if self.is_improved(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop