"""
This script saves precision and recall values with different confidence rates
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_p_r(model_path, classes, output_path):
    model = YOLO(model_path)

    confs = np.arange(0.01, 1.01, 0.01)
    df = pd.DataFrame({'Confidence': confs})

    for cl in classes:
        precisions, recalls = [], []

        for conf in confs:
            metrics = model.val(conf=conf, classes=cl)
            recalls.append(metrics.box.mr)
            precisions.append(metrics.box.mp)

        df[f'Precision_{cl}'] = precisions
        df[f'Recall_{cl}'] = recalls

    df.to_csv(output_path, index=False)
    return df


def plot_p_r(dfs, names, classes, output_path):
    for n, cl in enumerate(classes):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for k, df in enumerate(dfs):
            ax1.plot(df['Confidence'], df[f'Precision_{cl}'], color=n, label=f'{names[k]}')
            ax2.plot(df['Confidence'], df[f'Recall_{cl}'], color=n, label=f'{names[k]}')
        ax1.set_xlabel('Confidence')
        ax2.set_xlabel('Confidence')
        ax1.set_ylabel('Precision')
        ax2.set_ylabel('Recall')
        fig.title(f'Class: {cl}')

        fig.savefig(f'{output_path}test_{cl}.png')


# df1 = get_p_r('runs/detect/with_lamps_yolov8s_v8/weights/best.pt', range(2), 'test/confpr.csv')
df2 = get_p_r('best.pt', range(2), 'test/confpr.csv')

plot_p_r([df1,df2], ['lamps', '50e'], [0,1],'test/')

