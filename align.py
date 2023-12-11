import pandas as pd
import sentence_transformers as st
import numpy as np
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                    help='''Model name, models available at:
                            https://www.sbert.net/docs/pretrained_models.html''')
parser.add_argument('--lag', type=int, default=1,
                    help= 'Alignment (e.g., 1 computes for previous turn''')
parser.add_argument('--fname', type=str, default=None,
                    help= 'filename in data folder to read in')
parser.add_argument('--pair-type', type=str, default='true',
                    help= '"surrogate" or "true" pairs')


def main(model_id, lag, fname, pair_type, short_output=False):
    ''' Full pipeline for alignment extraction '''
    # Define outpath
    short_str = 'short' if short_output is True else ''
    OUTPATH = Path('outputs')
    OUTPATH.mkdir(exist_ok=True)
    outname = f"{fname.split('.')[0]}_lag-{lag}_model-{model_id}{short_str}.txt"
    outfile = OUTPATH / outname

    # Read in and import
    print('*** Preprocessing data ***')
    DPATH = Path('data') / fname
    sep = ',' if str(DPATH).endswith('csv') else '\t'
    data = pd.read_csv(str(DPATH), sep=sep)

    # Defined columns
    id_col = 'ChildID' if 'ChildID' in data.columns else 'ID'
    speakers = data['Speaker'].unique()
    child_id = 'Child' if 'Child' in speakers else 'CHI'
    caregiver_id = list(set(speakers) - set([child_id]))[0]

    # Compute expected size
    exp_size = (data.groupby([id_col,
                              'Visit'])['Transcript'].count() - lag).clip(lower=0).sum()

    # Get lagged time series
    if pair_type != 'surrogate':
        data = data.sort_values(by=[id_col, 'Visit', 'Turn']).reset_index()
    lagged = data.shift(lag)
    for c in ['Transcript', 'Visit', 'Speaker', id_col]:
        data[f'lagged_{c}'] = lagged[c]
    data.dropna(subset=['lagged_Transcript'], inplace=True)
    data = data[(data['Visit']==data['lagged_Visit']) &
                (data[id_col]==data[f'lagged_{id_col}'])]
    assert data.shape[0] ==  exp_size

    # Define model and similarity function
    print('*** Preparing SentenceBERT model ***')
    model = st.SentenceTransformer(f'{model_id}')
    def _get_encoding(row):
        enc_0 = model.encode(row['Transcript'].tolist())
        enc_1 = model.encode(row['lagged_Transcript'].tolist())
        sim = st.util.cos_sim(enc_0, enc_1).numpy()
        return sim[np.diag_indices(sim.shape[0])].round(4)

    # Extract alignment
    print('*** Extracting alignment ***')
    grouper = data.groupby([id_col, 'Visit'])
    encoded = grouper.apply(_get_encoding).reset_index().explode(0)[0]
    data['SemanticAlignment'] = encoded.tolist()

    # Add metadata and remove spurious pairs
    conditions = [
    ((data['Speaker']==caregiver_id) & (data['lagged_Speaker']==caregiver_id)),
    ((data['Speaker']==caregiver_id) & (data['lagged_Speaker']==child_id)),
    ((data['Speaker']==child_id) & (data['lagged_Speaker']==child_id)),
    ((data['Speaker']==child_id) & (data['lagged_Speaker']==caregiver_id)),
    ]
    choices = ["caregiver2caregiver",
               "caregiver2child",
               "child2child",
               "child2caregiver"]
    data['AlignmentType'] = np.select(conditions, choices)
    data['Lag'] = lag
    data['ModelId'] = model_id

    # Remove redundant columns
    if short_output:
        data = data[[id_col, 'Visit', 'Turn',
                    'Lag', 'ModelId',
                    'SemanticAlignment',
                    'AlignmentType']]
    data.to_csv(str(outfile), sep='\t', index=False)


if __name__=='__main__':
    args = parser.parse_args()
    main(args.model, args.lag, args.fname, args.pair_type)
