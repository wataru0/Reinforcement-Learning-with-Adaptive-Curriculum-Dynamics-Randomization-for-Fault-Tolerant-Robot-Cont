#!/bin/sh

# LD_PRELOAD= python eval.py --agent_id=baseline-v2
# LD_PRELOAD= python eval.py --agent_id=udr
# LD_PRELOAD= python eval.py --agent_id=cdr-v1
# LD_PRELOAD= python eval.py --agent_id=cdr-v1_
# LD_PRELOAD= python eval.py --agent_id=cdr-v2
# LD_PRELOAD= python eval.py --agent_id=cdr-v2_
# LD_PRELOAD= python eval.py --agent_id=lcdr-v1
# LD_PRELOAD= python eval.py --agent_id=lcdr-v2
# # LD_PRELOAD= python eval.py --agent_id=acdr_prot
# # LD_PRELOAD= python eval.py --agent_id=acdr_prot_softmax_t=1
# LD_PRELOAD= python eval.py --agent_id=acdr_prot_bayes
# LD_PRELOAD= python eval.py --agent_id=acdr_prot_bayes_4e6
# # LD_PRELOAD= python eval.py --agent_id=baseline

# LD_PRELOAD= python eval.py --agent_id=baseline_4e6
# LD_PRELOAD= python eval.py --agent_id=cdr-v1_16e6
# LD_PRELOAD= python eval.py --agent_id=cdr-v2_16e6

# LD_PRELOAD= python eval.py --agent_id=cdr-v1_bf50
# LD_PRELOAD= python eval.py --agent_id=cdr-v2_bf50

# LD_PRELOAD= python eval.py --agent_id=lcdr-v2_4e6
# LD_PRELOAD= python eval.py --agent_id=cdr-v1_bf10
# LD_PRELOAD= python eval.py --agent_id=cdr-v2_bf10

# generalizationのための評価
# python eval.py --agent_id=baseline_4e6 --type=gene
# python eval.py --agent_id=udr --type=gene
# python eval.py --agent_id=lcdr-v1 --type=gene
# python eval.py --agent_id=lcdr-v2_4e6 --type=gene
# python eval.py --agent_id=cdr-v1_bf10 --type=gene
# python eval.py --agent_id=cdr-v2_bf10 --type=gene
# python eval.py --agent_id=acdr_prot_bayes_4e6 --type=gene

# LD_PRELOAD= python eval.py --agent_id=cdr-v1_bf50_8e6
# LD_PRELOAD= python eval.py --agent_id=cdr-v2_bf50_8e6
# LD_PRELOAD= python eval.py --agent_id=cdr-v1_bf50_16e6
# LD_PRELOAD= python eval.py --agent_id=cdr-v2_bf50_16e6
# python eval.py --agent_id=cdr-v1_bf50_8e6 --type=gene
# python eval.py --agent_id=cdr-v2_bf50_8e6 --type=gene
# python eval.py --agent_id=cdr-v1_bf50_16e6 --type=gene
# python eval.py --agent_id=cdr-v2_bf50_16e6 --type=gene

# python eval.py --agent_id=lcdr-v2 --type=gene

# LD_PRELOAD= python eval.py --agent_id=acdrb_4e6_maximizeFlag=True
# python eval.py --agent_id=acdrb_4e6_maximizeFlag=True --type=gene

# kを固定したもの
# LD_PRELOAD= python eval.py --agent_id=k00
# LD_PRELOAD= python eval.py --agent_id=k04
# LD_PRELOAD= python eval.py --agent_id=k08
# LD_PRELOAD= python eval.py --agent_id=k12

# python eval.py --agent_id=k00 --type=gene
# python eval.py --agent_id=k04 --type=gene
# python eval.py --agent_id=k08 --type=gene
# python eval.py --agent_id=k12 --type=gene

LD_PRELOAD= python eval.py --agent_id=k08_2
python eval.py --agent_id=k08_2 --type=gene