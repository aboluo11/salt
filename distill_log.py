import re
from collections import defaultdict
import pandas as pd
import numpy as np


def get_data():
    tsfms = defaultdict(dict)
    with open('log', 'r') as f:
        log = f.read()
    chunks = log.split('----------------------------\n')
    ep_partten = re.compile(r'(?<=epoch=)\d+')
    score_partten = re.compile(r'(?<=best metric: )\d+\.\d+')
    for chunk in chunks:
        lines = chunk.split('\n')[:-1]
        epoch = int(ep_partten.search(lines[0]).group())
        score = float(score_partten.search(lines[-1]).group())
        tsfm = '\n'.join(lines[1:-2])
        tsfms[tsfm][epoch] = score
    return tsfms


def to_txt(tsfms):
    with open('temp.txt', 'w') as f:
        for tsfm, ep_scores in tsfms.items():
            f.write(tsfm + ':\n')
            for ep, score in ep_scores.items():
                f.write(f'  {ep}ep: {score}\n')
            f.write('\n')


def to_csv(tsfms):
    df = pd.DataFrame.from_dict(tsfms, orient='index')
    df.to_csv('record.csv')


tsfms = get_data()
to_txt(tsfms)
to_csv(tsfms)
