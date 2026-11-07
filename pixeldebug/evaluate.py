"""Detection metrics against target-element boxes, not semantic classifier accuracy."""
import argparse
import json
from pathlib import Path
from pixeldebug.vision import compare, decode_image


def iou(a, b):
    intersection = max(0, min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x'])) * max(0, min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y']))
    return intersection / max(a['width']*a['height']+b['width']*b['height']-intersection, 1)


def evaluate(root: Path, split='test'):
    rows = [json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['split'] == split]
    if not rows:
        raise ValueError(f'No examples in {split}; generate more template families')
    hits, boxes, overlaps = 0, 0, []
    for row in rows:
        result = compare(decode_image((root/row['reference']).read_bytes()), decode_image((root/row['defective']).read_bytes()))
        values = [iou(d['bbox'], row['bbox']) for d in result['defects']]
        best = max(values, default=0)
        overlaps.append(best)
        hits += best >= .5
        boxes += len(values)
    return {'split': split, 'examples': len(rows), 'target_recall_iou_0.5': hits/len(rows),
            'mean_best_iou': sum(overlaps)/len(rows), 'predicted_regions': boxes,
            'note': 'One target per example; boxes may fragment or include reflow. No semantic accuracy or real-world generalization claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--split', default='test', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.root, args.split), indent=2))
