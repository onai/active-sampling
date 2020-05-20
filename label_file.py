'''
'''

import pickle
import sys
import joblib
import json

def load_remove_set(f):
    the_set = []
    with open(f) as handle:
        for new_line in handle:
            the_set.append(' '.join(new_line.split()[1:]))

    return the_set

if __name__ == '__main__':
    filename = sys.argv[1]
    clf = joblib.load(sys.argv[2])
    dest = sys.argv[3]

    if len(sys.argv) > 4:
        remove_set = load_remove_set(sys.argv[4])
    else:
        remove_set = []

    all_lines = []
    
    with open(filename) as handle:
        for new_line in handle:
            tokens = new_line.split()

            if ' '.join(tokens) in remove_set:
                continue

            if len(tokens) > 100:
                continue

            all_lines.append(' '.join(tokens))


    import operator
    
    probabilities = clf.predict_proba(all_lines)

    items = []
    
    with open(dest, 'w') as handle:
        for i, line in enumerate(all_lines):
            items.append((line, probabilities[i,1]))

        sorted_items = sorted(items, key=operator.itemgetter(1), reverse=True)

        for k, v in sorted_items:
            handle.write(
                json.dumps(
                    {
                        'text': k,
                        'prob': v,
                        'label': 1,
                        'category': 0
                    }
                )
            )
            handle.write('\n')
