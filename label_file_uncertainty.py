'''
'''

import pickle
import sys
import joblib
import json

if __name__ == '__main__':
    filename = sys.argv[1]
    clf = joblib.load(sys.argv[2])
    dest = sys.argv[3]
    remove = sys.argv[4]

    all_lines = []

    remove_set = set([x.strip() for x in open(remove).readlines()])
    
    with open(filename) as handle:
        for new_line in handle:
            
            tokens = new_line.strip().split()

            if len(tokens) > 100:
                continue

            sent = ' '.join(tokens)

            if sent in remove_set:
                continue

            all_lines.append(sent)


    import operator
    
    probabilities = clf.predict_proba(all_lines)

    items = []
    
    with open(dest, 'w') as handle:
        for i, line in enumerate(all_lines):
            items.append((line, probabilities[i,1]))

        sorted_items = sorted(items, key=lambda x: abs(0.5 - x[1]))

        for k, v in sorted_items:
            handle.write(
                json.dumps(
                    {
                        'text': k,
                        'label': 0
                    }
                )
            )
            handle.write('\n')
