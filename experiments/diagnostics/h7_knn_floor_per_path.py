"""H7 follow-up: per-test-path kNN-floor breakdown + comparison to our 9.1 m result."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from h7_knn_floor import load_path, build_matrix, knn_predict, TRAIN, TEST, FILL  # noqa


def main():
    X_tr, y_tr, cols = build_matrix(TRAIN)
    print(f'train: {X_tr.shape}')
    out = {'per_test_path': {}}
    err_total = []
    for pid in TEST:
        rssi, xy, ap_cols = load_path(pid)
        # align AP space
        mapping = {c: i for i, c in enumerate(ap_cols)}
        Q = np.full((rssi.shape[0], len(cols)), FILL, dtype=np.float32)
        for j, c in enumerate(cols):
            src = mapping.get(c)
            if src is not None:
                Q[:, j] = rssi[:, src]
        for k in (1, 3, 9):
            pred, _ = knn_predict(X_tr, y_tr, Q, k)
            err = np.linalg.norm(pred - xy, axis=1)
            out['per_test_path'].setdefault(f'k={k}', {})[f'path_{pid}'] = {
                'MAE_m': float(err.mean()),
                'n': int(len(err)),
                'median_m': float(np.median(err)),
            }
            if k == 9:
                err_total.append(err)
    flat = np.concatenate(err_total)
    out['k=9_overall'] = {
        'MAE_m': float(flat.mean()),
        'median_m': float(np.median(flat)),
        'p90_m': float(np.percentile(flat, 90)),
    }
    print(json.dumps(out, indent=2))
    Path('x:/navlori-fusion/experiments/diagnostics/h7_knn_floor_per_path.json').write_text(
        json.dumps(out, indent=2)
    )


if __name__ == '__main__':
    main()
