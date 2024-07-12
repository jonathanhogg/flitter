"""
Extract coverage information for making the README badge
"""

import colorsys
import json

with open('coverage.json') as f:
    data = json.load(f)
    percent = data['totals']['percent_covered']
    r, g, b = colorsys.hsv_to_rgb(max(0, percent/100-0.5)*2/3, 0.9, 0.7)
    color = (int(round(r*16)) << 8) + (int(round(g*16)) << 4) + int(round(b*16))
    print(f'status={percent:.0f}%')
    print(f'color={color:03X}')
