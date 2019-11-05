from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import pandas as pd


def main():
    y_true = pd.Series(
        ["not mafia", "not mafia", "mafia", "not mafia", "mafia", 
        "not mafia", "not mafia", "mafia", "not mafia", "not mafia"]
        )
    y_pred = pd.Series(
        ["mafia", "mafia", "not mafia", "not mafia", "mafia", 
        "not mafia", "not mafia", "mafia", "not mafia", "not mafia"]
        )
    
    print("Confusion Matrix :\n",pd.crosstab(y_pred, y_true, rownames=['Predicted'], colnames=['Actual'], margins=True))

        
    print("\n\nClassification report : \n", classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()