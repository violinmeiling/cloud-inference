import pandas as pd
from sklearn.metrics import classification_report

def print_classification_report(csv_path):
    df = pd.read_csv(csv_path)

    # Map labels from Y/N to 1/0 for sklearn compatibility
    label_map = {'Y': 1, 'N': 0}
    y_true = df['electric_car'].map(label_map)
    y_pred = df['model_electric_car'].map(label_map)

    # Generate and print classification report
    report = classification_report(y_true, y_pred, target_names=['N', 'Y'])
    print(report)

if __name__ == "__main__":
    print_classification_report("./comparison_report.csv")
