import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score


class Evaluate:
    def calculate_confusion_matrix(self, truth, pred, model):
        # cf = confusion_matrix(truth, pred)
        tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
        print("--- Model ", model, " ---")
        print("Confusion Matrix (tn, fp, fn, tp): ", tn, fp, fn, tp)

        acc = accuracy_score(truth, pred)
        recall = recall_score(truth, pred)
        prec = precision_score(truth, pred)
        f1 = f1_score(truth, pred)
        print("-- Metrics --")
        print("> Accuracy: ", acc)
        print("> Precision: ", prec)
        print("> Recall: ", recall)
        print("> F1: ", f1)

    def plot_roc_curve(self, y_test, y_score, model):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr)
        plt.title(model + " ROC Curve")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()