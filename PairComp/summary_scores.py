import os
import json
import argparse

parser = argparse.ArgumentParser(description="evaluate_images")

parser.add_argument("--tgtpath", type=str,help="the path of the score json", default='prompt/score.json')

args = parser.parse_args()

data = json.load(open('prompt/PairComp.json'))
scores = json.load(open(args.tgtpath))

overall = [[], []]
color = [[], []]
counting = [[], []]
position = [[], []]
style_tone = [[], []]
text = [[], []]

for key in scores.keys():
    scorelist = scores[key]
    sa = 0.0
    sg = 1.0
    for s in scorelist:
        sa += s['score']
        sg *= s['score']
    sa = sa / len(scorelist)
    sg = sg ** (1.0/len(scorelist)) 
    category = data[int(key)]['category'].lower()
    if 'overall' in category:
        overall[0].append(sa)
        overall[1].append(sg)
    if 'color' in category:
        color[0].append(sa)
        color[1].append(sg)
    if 'counting' in category:
        counting[0].append(sa)
        counting[1].append(sg)
    if 'position' in category:
        position[0].append(sa)
        position[1].append(sg)
    if 'style' in category:
        style_tone[0].append(sa)
        style_tone[1].append(sg)
    if 'text' in category:
        text[0].append(sa)
        text[1].append(sg)

assert len(overall[0])==250
assert len(color[0])==50
assert len(counting[0])==50
assert len(position[0])==50
assert len(style_tone[0])==100
assert len(text[0])==50

print(f'*******score for {args.tgtpath}***********')
print('Overall appearance:')
sa1 = sum(overall[0])/len(overall[0])
sg1 = sum(overall[1])/len(overall[1])
print(f'sa:{sa1}', f'sg:{sg1}')

print('Color:')
sa2 = sum(color[0])/len(color[0])
sg2 = sum(color[1])/len(color[1])
print(f'sa:{sa2}', f'sg:{sg2}')

print('Counting:')
sa3 = sum(counting[0])/len(counting[0])
sg3 = sum(counting[1])/len(counting[1])
print(f'sa:{sa3}', f'sg:{sg3}')

print('Position:')
sa4 = sum(position[0])/len(position[0])
sg4 = sum(position[1])/len(position[1])
print(f'sa:{sa4}', f'sg:{sg4}')

print('Style & Tone:')
sa5 = sum(style_tone[0])/len(style_tone[0])
sg5 = sum(style_tone[1])/len(style_tone[1])
print(f'sa:{sa5}', f'sg:{sg5}')

print('Text:')
sa6 = sum(text[0])/len(text[0])
sg6 = sum(text[1])/len(text[1])
print(f'sa:{sa6}', f'sg:{sg6}')

print('\nAverage:')
sa_avg = (sa1+sa2+sa3+sa4+sa5+sa6)/6
sg_avg = (sg1+sg2+sg3+sg4+sg5+sg6)/6
print(f'sa:{sa_avg}', f'sg:{sg_avg}')

