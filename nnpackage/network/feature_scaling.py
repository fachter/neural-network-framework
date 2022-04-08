class StandardScaler:
    def __init__(self, mean=1., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, f):
        self.mean = f.mean(axis=0)
        self.std = f.std(axis=0)
        self.std[self.std == 0] = 0.0000001

    def transform(self, f):
        f = f - self.mean
        f = f / self.std
        return f

    def inverse_transform(self, f):
        f *= self.std
        f += self.mean
        return f


class NormalScaler:
    def __init__(self, min_value=1, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X):
        self.min_value = X.min(axis=0)
        self.max_value = X.max(axis=0) - self.min_value

    def transform(self, X):
        return (X - self.min_value) / self.max_value

    def inverse_transform(self, X):
        return X * self.max_value + self.min_value
