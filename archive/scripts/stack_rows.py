"""Scored, required rows of valid for the row stacks and their gates: current row X and its diff to the previous row
(what the runtime sees). Usage: stack_rows.py <out dir/> -> stackB_X.npy, stackB_D.npy (pipeline/regime_gates/proj_stack2.py)."""
import numpy as np, pyarrow.parquet as pq, sys
A = '/home/user/data/wnn/artifacts/'
m = (np.load(A + 'preds/valid_mask.npy') & np.load(A + 'preds/valid_need.npy')).astype(bool).reshape(-1, 20000)
f = pq.ParquetFile('/home/user/data/wnn/valid.parquet')
feat = [n for n in f.schema_arrow.names if n not in ('seq_ix','step_in_seq','need_prediction','is_scored','t0','t1')]
Xs, Ds = [], []
for g in range(f.metadata.num_row_groups):
    t = f.read_row_group(g, use_threads=False, columns=feat)
    X = np.stack([t.column(c).to_numpy() for c in feat], 1).astype(np.float32)
    D = np.vstack([np.zeros((1, X.shape[1]), np.float32), np.diff(X, axis=0)])
    Xs.append(X[m[g]]); Ds.append(D[m[g]])
np.save(sys.argv[1] + 'stackB_X.npy', np.concatenate(Xs)); np.save(sys.argv[1] + 'stackB_D.npy', np.concatenate(Ds))
print('rows', sum(len(x) for x in Xs))
