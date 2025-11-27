from sklearn.svm import OneClassSVM
import joblib

class OCSVMModel:
    def __init__(self):
        self.model = OneClassSVM(nu=0.02, kernel="rbf")

    def train(self, X):
        self.model.fit(X)

    def save(self, path):
        joblib.dump(self.model, path)
