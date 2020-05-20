import sys
import json

if __name__ == '__main__':
    f = sys.argv[1]

    with open(f) as handle:
        for new_line in handle:
            payload = json.loads(new_line)
            print(payload['label'], payload['text'])
