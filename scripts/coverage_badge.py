"""
Extract coverage information for making the README badge
"""

import json

with open('coverage.json') as f:
    data = json.load(f)
    percent = data['totals']['percent_covered']
    print(f'status={percent:.0f}%')
    if percent > 90:
        print('color=blue')
    if percent > 80:
        print('color=cyan')
    elif percent > 70:
        print('color=green')
    elif percent > 60:
        print('color=BA1')
    elif percent > 50:
        print('color=orange')
    else:
        print('color=red')
