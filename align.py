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
parser.add_argument('--pair-type', type=str, default='true',
                    help= '"surrogate" or "true" pairs')


def main(model_id, lag, pair_type):
    ''' Full pipeline for alignment extraction '''

    # Define outpath
    OUTPATH = Path('outputs')
    OUTPATH.mkdir(exist_ok=True)
    if pair_type == 'true':
        outfile = OUTPATH / f'{pair_type}_lag-{lag}_model-{model_id}.txt'
    else:
        outfile = OUTPATH / f'{pair_type}_model-{model_id}.txt'

    # Read in and import
    print('*** Preprocessing data ***')
    if pair_type == 'true':
        DPATH = Path('data') / 'transcripts.txt'
    elif pair_type == 'surrogate':
        DPATH = Path('data') / 'surrogates.txt'
    else:
        raise ValueError('''pair_type should be "true" or "surrogate"''')
    data = pd.read_csv(str(DPATH), sep='\t')

    # Compute expected size
    exp_size = (data.groupby(['ChildID',
                              'Visit'])['Transcript'].count() - lag).sum()

    # Get lagged time series
    if pair_type == 'true':
        data = data.sort_values(by=['ChildID', 'Visit', 'Turn']).reset_index()
    data.drop([f'V{i}' for i in range(2,302)], axis=1, inplace=True)
    lagged = data.shift(lag)
    for c in ['Transcript', 'Visit', 'Speaker', 'ChildID']:
        data[f'lagged_{c}'] = lagged[c]
    data.dropna(subset=['lagged_Transcript'], inplace=True)
    data = data[(data['Visit']==data['lagged_Visit']) &
                (data['ChildID']==data['lagged_ChildID'])]
    assert data.shape[0] ==  exp_size

    # Define model and similarity function
    print('*** Preparing SentenceBERT model ***')
    model = st.SentenceTransformer(model_id)
    def _get_encoding(row):
        enc_0 = model.encode(row['Transcript'].tolist())
        enc_1 = model.encode(row['lagged_Transcript'].tolist())
        sim = st.util.cos_sim(enc_0, enc_1).numpy()
        return sim[np.diag_indices(sim.shape[0])].round(4)

    # Extract alignment
    print('*** Extracting alignment ***')
    grouper = data.groupby(['ChildID', 'Visit'])
    encoded = grouper.apply(_get_encoding).reset_index().explode(0)[0]
    data['SemanticAlignment'] = encoded.tolist()

    # Add metadata and remove spurious pairs
    conditions = [
    (data['Speaker']=='MOT' & data['lagged_Speaker']=='MOT'),
    (data['Speaker']=='MOT' & data['lagged_Speaker']=='CHI'),
    (data['Speaker']=='CHI' & data['lagged_Speaker']=='CHI'),
    (data['Speaker']=='CHI' & data['lagged_Speaker']=='MOT'),
    ]
    choices = ["caregiver2caregiver", "caregiver2child",
                "child2child", "child2caregiver"]
    data['AlignmentType'] =  np.select(conditions, choices)
    data['Lag'] = lag
    data['ModelId'] = model_id

    # Remove redundant columns
    data = data[['ChildID', 'Visit', 'Turn',
                 'Lag', 'ModelId',
                 'SemanticAlignment',
                 'AlignmentType']]
    data.to_csv(str(outfile), sep='\t', index=False)


if __name__=='__main__':
    args = parser.parse_args()
    main(args.model, args.lag, args.pair_type)
