'''
'''

import json
import numpy as np
import sys

if __name__ == '__main__':
    f = sys.argv[1]
    remove_set = sys.argv[2]
    removes = []

    with open(remove_set) as handle:
        for new_line in handle:
            x = ' '.join(new_line.strip().split()[1:])
            removes.append(x)

    all_sents = set([])
            
    with open(f) as handle:
        for new_line in handle:
            new_line = new_line.strip()
            if new_line in removes:
                continue

            toks = new_line.split()

            if len(toks) > 100:
                continue

            all_sents.add(new_line)

    
    samples = np.random.choice(list(all_sents), size=300, replace=False)

    for x in samples:
        print(
            json.dumps(
                {
                    'text': x,
                    'label': 0
                }
            )
        )
