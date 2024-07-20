import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def calculate_metrics(true_seqs, pred_seqs):
    assert len(true_seqs) == len(pred_seqs), "The number of true sequences and predicted sequences must be the same"

    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    exact_matches = 0
    amino_acid_coverages = []
    mrrs = []
    y_true_all = []
    y_pred_all = []

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        true_chars = list(true_seq)
        pred_chars = list(pred_seq)
        common_length = min(len(true_chars), len(pred_chars))

        # Calculate precision, recall, f1 score, accuracy
        y_true = [c for c in true_chars[:common_length]]
        y_pred = [c for c in pred_chars[:common_length]]

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        precisions.append(precision_score(y_true, y_pred, average='micro', zero_division=0))
        recalls.append(recall_score(y_true, y_pred, average='micro', zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, average='micro', zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))

        # Calculate exact match ratio
        if true_seq == pred_seq:
            exact_matches += 1

        # Calculate amino acid coverage
        coverage = sum([1 for i in range(common_length) if true_chars[i] == pred_chars[i]]) / len(true_chars)
        amino_acid_coverages.append(coverage)

        # Calculate mean reciprocal rank
        ranks = [i+1 for i in range(common_length) if true_chars[i] == pred_chars[i]]
        if ranks:
            mrrs.append(1 / ranks[0])
        else:
            mrrs.append(0)

    results = {
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1 Score': np.mean(f1_scores),
        'Accuracy': np.mean(accuracies),
        'Exact Match Ratio': exact_matches / len(true_seqs),
        'Amino Acid Coverage': np.mean(amino_acid_coverages),
        'Mean Reciprocal Rank': np.mean(mrrs)
    }

    # Print results
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Plot results
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values())
    ax.set_ylabel('Score')
    ax.set_title('Peptide Sequencing Accuracy Metrics')
    plt.xticks(rotation=45)
    plt.show()

    # Confusion Matrix
    labels = list(set(y_true_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.show()

    # AUC-ROC Curve
    y_true_bin = label_binarize(y_true_all, classes=labels)
    y_pred_bin = label_binarize(y_pred_all, classes=labels)

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(len(labels)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Example usage:
true_sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDFGIKLMNPQRSTVWY"]
predicted_sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDFGIKLMMNPQRSTVWY"]

calculate_metrics(true_sequences, predicted_sequences)
