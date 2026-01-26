# Prosody Project: Full Pipeline (Python + MATLAB)

This guide explains how to run the **complete prosody extraction pipeline**, from raw CommonVoice data through synthetic prosody generation and transformer training.

---

## Step 1: Upload Audio Data

> Organize your raw audio datasets for training.

**Assumption:** Audio data comes from Mozilla CommonVoice.

1. Upload all audio datasets into a single folder within the project directory.  
2. Name each dataset by its language abbreviation (e.g., `english` → `en`, `spanish` → `es`).  

---

## Step 2: Prepare Files for MATLAB Processing

> Generate `.txt` file lists that MATLAB will use as input.

1. Open **`matlab/get_data.py`** and 
   1. set the `start` field as the current directory of audio
   2. set the `destination` field as where you want the data to go
   3. set the `language` field as the location of the language currently
2. Run the python file, it should create a `data\lang.txt` file and update `config.json`

---

## Step 3: Generate Synthetic Audio (MATLAB)

> Run MATLAB scripts to produce prosody-only audio clips.

1. Open **`run_file.sh`** and list the target languages to process. 
2. Make the script executable:

   ```bash
   chmod +x run_file.sh
   ```

3. Submit it as a batch job:

   ```bash
   sbatch run_file.sh
   ```

4. Progress logs will appear in the `logs/` folder.

---
## Step 4: Update config file
> Write audio location in config file so scripts knows where audio is

1. Go into `config.py` and under `AUDIO_LOCATION` place the root of the matlab processed audio files.

---


## Step 5: Clean the Dataset

> Generate a length summary for all audio files to assist with later filtering.

This step builds a Pandas DataFrame recording the duration and path of each file.  
It typically takes **~40 minutes per language**, so run this once before setting up the ML stage.

1. Navigate to `scripts/lengths.py`. 
2. In the `if __name__ == "__main__":` block, list all language directories in the `languages` list. 
   1. Set languages as the list of languages you want processes
   1. Update the location to where the prosody parsed files are
3. From the `src` directory, run:

   ```bash
   PYTHONPATH=. python scripts/lengths.py
   ```

4. A file named **`lengths`** will be created inside each language’s directory.

---

## Step 5: View the Training Dataset
> Figure out how to adjust weights

1. Go into `plots/unique_speakers` and `plots/speakers.ipynb` specify the directories where the matlab files are run

1. Use the graphs to make an informed decision about how much weight to place for each audio length in `config.py`.

---

## Step 6: Generate Spectrograms

> Convert prosody-only audio into spectrograms for model training.

This will create a files under `lang/spect/` that contains each spectrogram batch in processed audio

1. Open **`scripts/make_spect.py`** and ensure the `location` variable matches the `save_directory`.  
2. Optionally, adjust parameters such as:
   - `n_fft`
   - `hop_length`
   - `sr`
   - the spectrogram function itself  
3. Run the script:

   ```bash
   PYTHONPATH=. python scripts/make_spect.py
   ```

---

## Step 8: Train the Transformer Model

> Train the language-detection transformer on the generated spectrograms.

1. In **`scripts/train.py`**, update:
   - `origin` → the directory containing spectrograms  
   - `base` → the directory where outputs (checkpoints, logs, etc.) will be saved  
2. Run training:

   ```bash
   PYTHONPATH=. python scripts/train.py
   ```

---

## Step 9: Evaluate the Model

> Review accuracy, confusion matrices, and performance metrics.

1. Open the **`review_model.ipynb`** notebook.  
2. Set:
   - `base` → the directory where the trained model is saved  
   - `name` → a label for the model being evaluated  
3. Run all cells to generate the evaluation results (confusion matrices, accuracy, etc.).

---

### Notes

- For reproducibility, keep all directory paths relative to the project root.  
- Use consistent sample rates (default `16 kHz`) across all audio.  
- **Script names:** If your repo uses `load_for_sythetic.py` (typo), adjust the command in Step 3 accordingly.  
- You can visualize spectrograms or audio outputs in the Jupyter notebooks under `notebooks/play/`.

---

**End of Pipeline**
