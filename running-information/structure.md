# This Notebook Discussess the Structure of Codebase

## Flow of files
```mermaid
graph LR
    A[Common-Voice Files] --> B[Prosody Filter] --> C[Model Information]
```

### Common Voice Files
> This directory contains all languages downloaded from commonvoice with no modifications to them

### Prosody Filter
> This directory contains all the prosody filtered files, along with information on audio file length

#### Breakdown

.
├── en
│   ├── clips
│   ├── custom
│   │   └── lengths.csv
│   └── validated.tsv
├── it
└── ja

### Model Information
> This directory contains all information involved in generating a model
- Spectrograms by language
- Dataset distribution
- Models trained on spectrograms

#### Folder Breakdown
.
├── en
│   └── spect
       └── test
          ├── Batch00001.pt
          ├── ...
          └── Batchxxxx.py 
       └── train
          ├── Batch00001.pt
          ├── ...
          └── Batchxxxx.py 
       └── validate
          ├── Batch00001.pt
          ├── ...
          └── Batchxxxx.pt 
├── it
├── ja
├── models
    ├── Model-1
    ├── ...
    └── Model-n
└── statistics
    ├── dataset.pkl
    ├── lengths.pkl
    └── speakers.pkl

### Organization

## Flow of creating a model
```mermaid
graph TB
    A[Common Voice .tar.gz] -- Upload into Server --> B(Place in directory)
    B  --> EN
    B  --> GE
    B  --> DE

    EN  -- Perform Filtering --> EN1[Augmented Data/EN]
    GE  -- Perform Filtering --> GE1[Augmented Data/GE]
    DE  -- Perform Filtering --> DE1[Augmented Data/DE]


    EN1 -- Run Dataset Collector --> C[Augmented Data]
    GE1 -- Run Dataset Collector --> C
    DE1 -- Run Dataset Collector --> C

    C -- Run Training Set Distribution --> D[View dataset distirbution]

    D -- Adjust Weights in Config --> E[Decide on Data Shape]

    E -- Create Spectrograms --> F

```