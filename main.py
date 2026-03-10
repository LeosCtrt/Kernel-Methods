import numpy as np
import pandas as pd
from kernels import LinearKernel, RBFKernel
from svm import MulticlassSVM

class StandardScaler:
    """DIY Standard Scaler (Mean=0, Variance=1)"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Add a tiny epsilon to prevent division by zero
        return (X - self.mean) / (self.std + 1e-8)

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

def main():
    print("1. Loading raw flat data...")
    # Keep the data completely flat this time (N, 3072)
    Xtr = np.array(pd.read_csv('data/Xtr.csv', header=None, sep=',', usecols=range(3072)))
    Xte = np.array(pd.read_csv('data/Xte.csv', header=None, sep=',', usecols=range(3072)))
    Ytr = np.array(pd.read_csv('data/Ytr.csv', sep=',', usecols=[1])).squeeze()

    print("2. Standardizing the features...")
    # This step is non-negotiable for raw pixel SVMs!
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    print("3. Training the Multiclass SVM with a Linear Kernel...")
    # A Linear Kernel is much safer to start with.
    # C=0.1 or C=1.0 is usually good for scaled linear data.
    kernel = LinearKernel() 
    classifier = MulticlassSVM(kernel=kernel, C=1.0)
    
    classifier.fit(Xtr_scaled, Ytr)
    print("Training complete!")

    print("4. Predicting on the test data...")
    Yte_predictions = classifier.predict(Xte_scaled)
    
    # Let's print the unique predictions to verify we broke the "single class" curse!
    unique_preds, counts = np.unique(Yte_predictions, return_counts=True)
    print(f"\nPrediction Distribution (Class : Count):")
    for cls, count in zip(unique_preds, counts):
        print(f"Class {int(cls)} : {count}")

    print("\n5. Formatting submission file...")
    submission_dict = {'Prediction': Yte_predictions.astype(int)}
    dataframe = pd.DataFrame(submission_dict)
    dataframe.index += 1
    dataframe.to_csv('results/Yte_pred_linear_baseline.csv', index_label='Id')
    print("Success! Predictions saved.")

if __name__ == "__main__":
    main()