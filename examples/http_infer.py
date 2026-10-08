"""Call the policy service with one explicit recurrent session."""
import argparse
import json
import os
import urllib.request

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8765')
parser.add_argument('--observation', required=True)
parser.add_argument('--legal-count', type=int, required=True)
parser.add_argument('--session', required=True)
parser.add_argument('--keep-session', action='store_true')
args = parser.parse_args()
with np.load(args.observation, allow_pickle=False) as data:
    observation = {key: data[key].tolist() for key in data.files}

def post(endpoint, body):
    headers = {'Content-Type': 'application/json'}
    token = os.environ.get('YGO_API_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(args.url.rstrip('/') + endpoint,
                                     json.dumps(body).encode(), headers, method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)

try:
    print(json.dumps(post('/infer', {'session': args.session, 'legal_count': args.legal_count,
                                   'observation': observation}), indent=2))
finally:
    if not args.keep_session:
        post('/reset', {'session': args.session})
