# -*- coding: utf-8 -*-
"""submit
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import utils

import pdb

# detections
predict_1 = 'predict_1.csv'
predict_2 = 'predict_2.csv'

def main():
    df_1 = pd.read_csv(predict_1)
    df_2 = pd.read_csv(predict_2)

    final_result = []
    for img_name in df_1.filename:
        box1 = np.array(df_1[df_1.filename == img_name])[:, 1:]
        box2 = np.array(df_2[df_2.filename == img_name])[:, 1:]
        
        preds = np.vstack((box1, box2, box3))

        preds = preds[preds[:, -1].argsort()]
        box = np.mean(preds[:, 0:-2].astype(np.float64), axis=0)
        pred_cls = np.unique(preds[:, -2])
        sign = preds[-1][-2]
        if len(pred_cls) > 1:
            print(img_name, pred_cls)
        
        final_result.append(np.hstack((np.array(img_name), box, sign)))
    
    columns = ['filename','X1','Y1','X2','Y2','X3','Y3','X4','Y4','type']
    df = pd.DataFrame(final_result, columns=columns)
    df.to_csv('predict.csv', index=False)

if __name__ == '__main__':
    main()
