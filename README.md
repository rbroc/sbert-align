# sbert-align
Compute parent-child alignment using SentenceBERT.
Available models can be found at: https://www.sbert.net/docs/pretrained_models.html

### Usage
1. Create a virtual environment (not necessary).

You can do so by typing:

``` python3 -m venv PATH_TO_ENV
source PATH_TO_ENV/bin/activate
Replace PATH_TO_ENV with path for virtual environment
```

2. Install requirements
```pip install -r requirements.txt```

3. Run the `align.py` script, e.g.,:
`python3 align.py --lag 1 --model all-mpnet-base-v2`.
For this to work, you need to have a `transcripts.tsv` file in the `data` folder.
Outputs are saved in `outputs` folder.

4. Deactivate once you're done
```deactivate```

### Instructions
- Sort by child, visit, turn. Compute cosine similarity among all adjacent utterances, marking if it's child2mot, or mot2child. Make sure not to cross visit boundaries

- Compute alignment with previous turn

- Data:
    - doc_id: child ID
    - visit: visit #
    - speaker: speaker
    - turn: does not discriminate between child and parent
    - lemmas: lemmatized transcript
    - PoS: parts of speech
    - Transcript: text
    - Turn; ChildID; Visit; Speaker (repeated)

- Output: ChildID, Visit, Turn, SemanticAlignment, AlignmentType.

- Surrogate pairs analysis: [Il secondo passo sarebbe fare lo stesso su un subset delle surrogate pairs. Le surrogate pairs sono qui: https://www.dropbox.com/sh/4zpghkadqogmq51/AADmFaZIveT-SVcKW-C0hhF5a?dl=0 (I file che cominciano con dS, separati per Gruppo diagnostic e visita). Secondo me prendere 30-50 childID at random da dS_ASD1_udpiped e 30-50 da dS_TD1_udpiped e poi analizzare solo quelli (anche nelle visite successive) dovrebbe bastare]