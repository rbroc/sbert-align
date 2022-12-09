import pandas as pd
import random
import glob
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sampled-ids', type=int, default=50,
                    help='''How many ids to sample''')

def main(n_ids):
    random.seed(42)

    # Get ids
    ref_ids = []
    for group in ['ASD', 'TD']:
        REF_PATH = Path('data') / 'raw' / f'dS_{group}1_udpiped.csv'
        ref_df = pd.read_csv(str(REF_PATH))
        ref_ids += random.sample(ref_df.ChildID.unique().tolist(),
                                 k=n_ids)
    with open(Path('data') / 'sampled_ids.txt', 'w') as fh:
        fh.write('\n'.join(ref_ids))

    # Process other surrogate pairs and reduce
    fs = glob.glob(str(Path('data') / 'raw' / '*'))
    dfs = []
    for i,f in enumerate(fs):
        print(f'Processing {i} out of {len(fs)}')
        df = pd.read_csv(f)
        df = df[df['ChildID'].isin(ref_ids)]
        dfs.append(df)
    outfile =  Path('data') / 'surrogates.txt'
    pd.concat(dfs, ignore_index=True).to_csv(str(outfile),
                                             sep='\t', 
                                             index=False)

if __name__=='__main__':
    args = parser.parse_args()
    main(args.sampled_ids)
