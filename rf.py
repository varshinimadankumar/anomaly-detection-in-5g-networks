from sklearn.ensemble import RandomForestClassifier
import joblib

class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)

    def train(self, X, y):
        self.model.fit(X, y)

    def save(self, path):
        joblib.dump(self.model, path)
