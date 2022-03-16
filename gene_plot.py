# 2022/1/19
# 各手法のgeneralizationを折れ線グラフでプロットするプログラム
# python eval.py --agent_id=@@ --type=gene　をした後に実行する！

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d # scipyのモジュールを使う
import argparse
from scipy import signal
import datetime as dt

# スプライン補間でスムージング
def spline_interp(in_x, in_y):
    out_x = np.linspace(np.min(in_x), np.max(in_x), np.size(in_x)*1000) # もとのxの個数より多いxを用意
    func_spline = interp1d(in_x, in_y, kind='cubic') # cubicは3次のスプライン曲線
    out_y = func_spline(out_x) # func_splineはscipyオリジナルの型

    return out_x, out_y

#移動平均でスムージング
def moving_avg(in_x, in_y):
    np_y_conv = np.convolve(in_y, np.ones(7)/float(7), mode='valid') # 畳み込む
    out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))

    return out_x_dat, np_y_conv

def main():
    # default
    # filename = ['baseline_4e6', 'udr', 'lcdr-v1', 'lcdr-v2_4e6', 'cdr-v1_bf10', 'cdr-v2_bf10', 'acdr_prot_bayes_4e6']
    ## filename = ['baseline_4e6', 'lcdr-v2_4e6', 'cdr-v2_bf10', 'acdr_prot_bayes_4e6']
    
    # variousK
    # filename = ['k00', 'k04', 'k08_2', 'k12', 'cdr-v2_bf10', 'acdr_prot_bayes_4e6']
    # filename = ['k00', 'k04', 'k08_2', 'k12', 'acdr_prot_bayes_4e6']
    
    # presentation
    filename = ['baseline_4e6', 'udr', 'lcdr-v2_4e6', 'acdr_prot_bayes_4e6']
    label = {
        'baseline-v2': 'Baseline',
        'baseline_4e6': 'Baseline',
        'udr': 'UDR',
        'lcdr-v1': 'LCDR_e2h',
        'lcdr-v2_4e6': 'LCDR_h2e',
        'cdr-v1_bf10': 'ACDR_e2h',
        'cdr-v2_bf10': 'ACDR_h2e',
        'acdr_prot_bayes_4e6': 'ACDRB',
        'k00': 'k=0.0', 
        'k02': 'k=0.2', 
        'k04': 'k=0.4', 
        'k06': 'k=0.6', 
        'k08': 'k=0.8',
        'k08_2': 'k=0.8', 
        'k12': 'k=1.2', 
        'k14': 'k=1.4',
    }

    # Create figure dir
    figdir = "./fig/gene/"
    os.makedirs(figdir,exist_ok=True)

    sns.set()
    plt.figure()
    fig, ax = plt.subplots()

    for file in filename:
        a = {}
        # if file == 'k00' or file =='k02' or file =='k04' or file =='k06' or file =='k08' or file =='k12' or file =='k14':
        #     for seed in range(1, 6):       
        #         a[seed] = np.load("./out/variousK/" + file + "/Baseline_Ant-v2_" + file + "_rewardForEachK_seed=" +  str(seed) + ".npy")
        # else:    
        #     for seed in range(1, 6):
        #         if file == 'lcdr-v2_4e6' and (seed == 3 or seed ==4):
        #             continue      
        #         a[seed] = np.load("./out/" + file + "/gene/seed=" +  str(seed) + ".npy")

        for seed in range(1, 6):
            if file == 'lcdr-v2_4e6' and (seed == 3 or seed ==4):
                continue      
            a[seed] = np.load("./out/" + file + "/gene/seed=" +  str(seed) + ".npy")

        # smoothing
        for i in range(1,6):
            # default is (a[i], 51, 3)
            # 21 3/ 31 3/ 41 3
            if file == 'lcdr-v2_4e6' and (i == 3 or i ==4):
                continue 
            a[i] = signal.savgol_filter(a[i],51,3)
        
        if file == 'lcdr-v2_4e6':
            col = np.linspace(0.0,1.0,100)
            a1 = a[1].reshape(100,1)
            a1 = np.insert(a1, 0, col, axis=1)
            a2 = a[2].reshape(100,1)
            a2 = np.insert(a2, 0, col, axis=1)
            a5 = a[5].reshape(100,1)
            a5 = np.insert(a5, 0, col, axis=1)
            aa = np.concatenate([a1,a2,a5])

        else:
            col = np.linspace(0.0,1.0,100)
            a1 = a[1].reshape(100,1)
            a1 = np.insert(a1, 0, col, axis=1)
            a2 = a[2].reshape(100,1)
            a2 = np.insert(a2, 0, col, axis=1)
            a3 = a[3].reshape(100,1)
            a3 = np.insert(a3, 0, col, axis=1)
            a4 = a[4].reshape(100,1)
            a4 = np.insert(a4, 0, col, axis=1)
            a5 = a[5].reshape(100,1)
            a5 = np.insert(a5, 0, col, axis=1)
            aa = np.concatenate([a1,a2,a3,a4,a5])
        # print(aa)
        # ndarray->pandas.DataFrame
        DF = pd.DataFrame(data=aa, columns=['k','AverageReward'], dtype='float')
        # print(DF)

        sns.lineplot(x="k", y="AverageReward", data=DF, ax=ax, label=label[file])

    ax.set_ylabel('Average Reward')
    ax.set_xlabel('k')
    ax.legend()


    # save fig
    plt.tight_layout()
    plt.tick_params(labelsize=10)
    ax.set_rasterized(True) 
    now = dt.datetime.now()
    # time = now.strftime("%Y%m%d-%H%M%S")
    # plt.savefig(figdir + "generalization_{}.png".format(time))
    plt.savefig(figdir + "generalization_.png")

if __name__=='__main__':
    main()