import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


class Evaluate:
    def calculate_confusion_matrix(self, truth, pred, model):
        # cf = confusion_matrix(truth, pred)
        tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
        print("--- Model ", model, " ---")
        print("Confusion Matrix (tn, fp, fn, tp): ", tn, fp, fn, tp)
        acc = (tp + tn) / (tp + fp + fn + tp)
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((prec * recall) / (prec + recall))
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