import pandas as pd
import json

def preprocess():
    df = pd.read_csv('../data/clean.csv')
    id_vars = ['ID', 'Visit', 'Task', 'Turn']
    df = df.groupby(id_vars, as_index=False).apply(lambda x: x.sort_values(by='StartTime').tail(1))
    df.to_csv('../data/clean.csv')

    # Add unique id to the data
    df['unique_id'] = df[id_vars].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Track which data should be nan-ed
    grouped = df.sort_values(by=id_vars).groupby(['ID',
                                                  'Visit', 
                                                  'Task'], 
                                                  as_index=False)
    no_previous = grouped.apply(lambda x: x[x['Turn']-x.shift()['Turn']>1]).unique_id # no previous turn_id available
    no_previous = no_previous.unique().tolist()
    df['no_2back'] = df['unique_id'].apply(lambda x: f"{'_'.join(x.split('_')[:3])}_{int(x.split('_')[-1])-1}" in no_previous) # if no 2back
    no_2back = df[df['no_2back']].unique_id.tolist()

    # Store info in a json file
    exclude = {'alignment_current_to_1back': no_previous,
               'alignment_1back_to_2back': no_previous + no_2back}
    with open('../data/exclude.json', 'w') as f:
        json.dump(exclude, f)


if __name__=='__main__':
    preprocess()
