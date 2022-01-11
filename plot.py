import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type', default=1, type=int)

    return parser.parse_args()

dir = "./out/" 
path = "eval/"
# agents = ["baseline-v2/", "udr/", "lcdr-v1/", "lcdr-v2/"]
# agents = ['baseline-v2/','acdr_prot_softmax_t=1/']
# agents = ['acdr_prot_bayes/']
agents = ["baseline_4e6/", "udr/", "lcdr-v1/", "lcdr-v2/", "cdr-v1_/", "cdr-v2_/", "acdr_prot_bayes_4e6/"]
labels = np.array(['Baseline', 'UDR', 'LCDR-e2h', 'LCDR-h2e', 'ACDR-e2h', 'ACDR-h2e', 'Bayes'])

figdir = './fig/'
os.makedirs(figdir, exist_ok=True)

plain_r = []
plain_re = []
plain_p = []
plain_pe = []

broken_r = []
broken_re = []
broken_p = []
broken_pe = []

for agent in agents:
    plain_reward = []
    plain_reward_error = []
    plain_progress = []
    plain_progress_error = []

    broken_reward = []
    broken_reward_error = []
    broken_progress = []
    broken_progress_error = []
    # default: range(1, 6)
    for seed in range(1, 6):
        # if agent == "cdr-v2_16e6/" and seed == 5:
        #     break
        # 各seedのplain rewardを取得
        json_open = open(dir + agent + path + "plain_result_seed={}.json".format(seed), 'r')
        json_load = json.load(json_open)
        plain_reward.append(json_load['episode_return']['mean'])
        plain_reward_error.append(json_load['episode_return']['std'])
        plain_progress.append(json_load['episode_forward_reward']['mean'])
        plain_progress_error.append(json_load['episode_forward_reward']['std'])

        # 各seedのbroken rewardを取得
        json_open = open(dir + agent + path + "broken_result_seed={}.json".format(seed), 'r')
        json_load = json.load(json_open)
        broken_reward.append(json_load['episode_return']['mean'])
        broken_reward_error.append(json_load['episode_return']['std'])
        broken_progress.append(json_load['episode_forward_reward']['mean'])
        broken_progress_error.append(json_load['episode_forward_reward']['std'])

    p_ave = sum(plain_reward)/len(plain_reward)
    p_error = sum(plain_reward_error)/len(plain_reward_error)
    p_p_ave = sum(plain_progress)/len(plain_progress)
    p_p_error = sum(plain_progress_error)/len(plain_progress_error)

    b_ave = sum(broken_reward)/len(broken_reward)
    b_error = sum(broken_reward_error)/len(broken_reward_error)
    b_p_ave = sum(broken_progress)/len(broken_progress)
    b_p_error = sum(broken_progress_error)/len(broken_progress_error)

    print(agent)
    print('plain ave: ', p_ave, p_error)
    print('broken ave: ', b_ave, b_error)
    plain_r.append(p_ave)
    plain_re.append(p_error)
    plain_p.append(p_p_ave)
    plain_pe.append(p_p_error)

    broken_r.append(b_ave)
    broken_re.append(b_error)
    broken_p.append(b_p_ave)
    broken_pe.append(b_p_error)

print('plain reward', plain_r)
print('plain progress',plain_p)

args = arg_parser()

# visualize average reward
if args.type == 0:
    plt.figure()
    plt.rcParams["figure.figsize"] = (10, 6)
    sns.set()
    fig, ax1 = plt.subplots()
    width = 0.35
    x = np.arange(len(plain_r))

    ax1.bar(x-width/2, plain_r, width=width, color='dodgerblue', align='center', label='plain', yerr=plain_re, capsize=5)
    ax1.bar(x+width/2, broken_r, width=width, color='mediumseagreen', align='center', label='broken', yerr=broken_re, capsize=5)

    ax1.set_ylabel("Average Reward")
    plt.tick_params(labelsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')

    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    plt.savefig(figdir + 'averageReward_{}.png'.format(time))
    plt.close('all')

    # visualize average progress
    plt.figure()
    plt.rcParams["figure.figsize"] = (10, 6)
    sns.set()
    fig, ax2 = plt.subplots()
    width = 0.35
    x = np.arange(len(plain_r))

    ax2.bar(x-width/2, plain_p, width=width, color='blue', align='center', label='plain', yerr=plain_pe, capsize=5)
    ax2.bar(x+width/2, broken_pe, width=width, color='green', align='center', label='broken', yerr=broken_pe, capsize=5)

    ax2.set_ylabel("Average Progress")
    plt.tick_params(labelsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(loc='upper left')

    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    plt.savefig(figdir + 'averageProgress_{}.png'.format(time))
    plt.close('all')

# # ==========
# visualize 2 figures
elif args.type == 1:
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (16, 6)

    sns.set()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    width = 0.35
    x = np.arange(len(plain_r))

    ax1.bar(x-width/2, plain_r, width=width, color='dodgerblue', align='center', label='plain', yerr=plain_re, capsize=3)
    ax1.bar(x+width/2, broken_r, width=width, color='mediumseagreen', align='center', label='broken', yerr=broken_re, capsize=3)

    ax1.tick_params(labelsize=7)
    ax1.set_ylabel("Average Reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')

    ax2.bar(x-width/2, plain_p, width=width, color='blue', align='center', label='plain', yerr=plain_pe, capsize=3)
    ax2.bar(x+width/2, broken_pe, width=width, color='green', align='center', label='broken', yerr=broken_pe, capsize=3)

    ax2.tick_params(labelsize=7)
    ax2.set_ylabel("Average Progress")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(loc='upper left')

    plt.tight_layout() # 追加
    plt.savefig(figdir + 'zemi.png')
    plt.close('all')