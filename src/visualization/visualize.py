import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from emg_decoder.src.visualization.key import Key

layout = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
          ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
          ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
          [' '],
          [chr(0)]]

if __name__ == '__main__':
    flat_ascii = Key(layout, ascii_code=0).flat_ascii
    order = np.argsort(flat_ascii)
    data_dir = "/Users/johnzhou/research/emg_decoder/data/processed/John-Zhou_2023-07-17_Open-Loop-Typing-Task"
    Xs = np.load(f"{data_dir}/aug/x_logits.npy")
    ys = np.load(f"{data_dir}/aug/y.npy")
    preds = np.argmax(Xs, axis=1)

    ys = [order[y] for y in ys]
    preds = [order[pred] for pred in preds]
    cm = confusion_matrix(ys, preds)
    cm = np.round(cm / cm.astype(np.float32).sum(axis=1), decimals=2)
    disp = ConfusionMatrixDisplay(cm, display_labels=[chr(c) for c in flat_ascii])
    fig, ax = plt.subplots(figsize=(15, 12.5))
    disp.plot(ax=ax)
    plt.show()
