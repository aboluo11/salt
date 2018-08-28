import re
from collections import defaultdict

with open('log', 'r') as f:
    log = f.read()

chunks = log.split('----------------------------\n')[:-1]

tsfms = defaultdict(dict)

ep_partten = re.compile(r'(?<=epoch=)\d+')
score_partten = re.compile(r'(?<=best metric: )\d+\.\d+')

for chunk in chunks:
    lines = chunk.split('\n')[:-1]
    epoch = ep_partten.search(lines[0]).group()
    score = score_partten.search(lines[-1]).group()
    tsfm = '\n'.join(lines[1:-2])
    tsfms[tsfm][epoch] = score

with open('temp.txt','w') as f:
    for tsfm, ep_scores in tsfms.items():
        f.write(tsfm+':\n')
        for ep, score in ep_scores.items():
            f.write(f'  {ep}ep: {score}\n')
        f.write('\n')