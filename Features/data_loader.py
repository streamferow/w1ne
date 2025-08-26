import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessing:

    def __init__(self, path):
        self.label_encoder = None
        self.scalar = None
        self.path = path
        self.df = None


    def load(self):
        self.df = pd.read_csv(self.path)
        return self


    def create_dataset(self, target_col = 'type'):
        X = self.df.drop(columns=[target_col]).copy()
        y = self.df[target_col].copy()

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.scalar = StandardScaler()
        X_scaled = self.scalar.fit_transform(X)
        print(X_scaled)

        return X_scaled, y_encoded


    @staticmethod
    def split_dataset(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
        return X_train, X_test, y_train, y_test

