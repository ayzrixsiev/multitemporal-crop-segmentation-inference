import numpy as np
import pandas as pd

"""
    mean Intersect over Union metric.
    computes the one versus all IoU for each class and returns the average.
    classes that do not appear in the provided set are not counted in the average.
    args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    returns:
        mean Iou (float)
"""


def mIou(y_true, y_pred, n_classes):
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed


"""
takes the confusion matrix and turns it into the actual report numbers.

the matrix is: row = what the pixel really was, column = what the model said.
so the diagonal is "guessed right" and everything else is a mistake.

for every class j we pull out three counts:

    tp = mat[j, j]              guessed this class, and it was this class
    fp = column j minus tp      said this class, but it was something else
    fn = row j minus tp         it was this class, but we said something else

from those three we get four ways of asking "how good is this class":

    precision = tp / (tp + fp)          when it says wheat, how often is it right
    recall    = tp / (tp + fn)          how much of the real wheat did it find
    f1        = balance of the two      one number instead of two
    iou       = tp / (tp + fp + fn)     how much the predicted blob overlaps
                                        the real field, punishes both mistakes

then the same numbers are averaged over classes in two different ways, and the
difference matters a lot:

    micro = throw every class into one bucket first, then divide once.
            counts pixels, so a huge class (meadow here) decides everything.

    macro = score each class on its own, then take a plain average.
            every class weighs the same, so a rare crop the model never finds
            drags the score down. this is the honest one, and the mIoU reported
            in papers is this one.

accuracy at the bottom is just diagonal / everything -- the share of pixels we
got right. it is the most forgiving number of all, which is why it sits well
above the macro iou.
"""


def confusion_matrix_analysis(mat):
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d["IoU"] = tp / (tp + fp + fn)
        d["Precision"] = tp / (tp + fp)
        d["Recall"] = tp / (tp + fn)
        d["F1-score"] = 2 * tp / (2 * tp + fp + fn)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall["micro_IoU"] = TP / (TP + FP + FN)
    overall["micro_Precision"] = TP / (TP + FP)
    overall["micro_Recall"] = TP / (TP + FN)
    overall["micro_F1-score"] = 2 * TP / (2 * TP + FP + FN)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall["MACRO_IoU"] = macro.loc["IoU"]
    overall["MACRO_Precision"] = macro.loc["Precision"]
    overall["MACRO_Recall"] = macro.loc["Recall"]
    overall["MACRO_F1-score"] = macro.loc["F1-score"]

    overall["Accuracy"] = np.sum(np.diag(mat)) / np.sum(mat)

    return per_class, overall
