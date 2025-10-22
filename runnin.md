# Prosody Project: Full Pipeline (Python + MATLAB)

This guide explains how to run the **complete prosody extraction pipeline**, from raw CommonVoice data through synthetic prosody generation and transformer training.

---

## Step 1: Upload Audio Data

> Organize your raw audio datasets for training.

**Assumption:** Audio data comes from Mozilla CommonVoice.

1. Upload all audio datasets into a single folder within the project directory.  
2. Name each dataset by its language abbreviation (e.g., `english` → `en`, `spanish` → `es`).  
3. Open **`config.py`** and update the variable **`AUDIO_LOCATION`** to match the path of the folder containing your uploaded audio files.

---

## Step 2: Clean the Dataset

> Generate a length summary for all audio files to assist with later filtering.

This step builds a Pandas DataFrame recording the duration and path of each file.  
It typically takes **~40 minutes per language**, so run this once before setting up the ML stage.

1. Navigate to `scripts/lengths.py`.  
2. In the `if __name__ == "__main__":` block, list all language directories in the `languages` list.  
3. From the `src` directory, run:

   ```bash
   PYTHONPATH=. python scripts/lengths.py
   ```

4. A file named **`lengths`** will be created inside each language’s directory.

---

## Step 3: Build the Training Dataset

> Use a greedy algorithm to select which audio samples to include.

1. In **`config.py`**, edit:
   - **`NUM_SPEAKERS`** to control how many clips per speaker to include (commonly `25`).
   - **`WEIGHTS`** to prioritize certain samples (see `training_distribution.ipynb` for guidance).  
2. From the `src` directory, run:

   ```bash
   PYTHONPATH=. python scripts/load_for_synthetic.py
   ```

3. The resulting dataset will be saved in `notebooks/play`.

---

## Step 4: Transfer Files to MATLAB

> Move your finalized dataset to the MATLAB project for prosody-only audio generation.

1. Copy all files from `notebooks/play` into the MATLAB project’s  
   `synthetic_audio/languages/` directory.

---

## Step 5: Prepare Files for MATLAB Processing

> Generate `.txt` file lists that MATLAB will use as input.

1. Open **`config.json`** and set the `save_directory` field to the destination for synthetic audio.  
2. Run the associated Jupyter notebook (e.g., `file.ipynb`) from start to finish.  
3. Once migration is complete, remove the `old_dir`, `new_dir`, and any `.replace()` lines.

---

## Step 6: Generate Synthetic Audio (MATLAB)

> Run MATLAB scripts to produce prosody-only audio clips.

1. Open **`run_file.sh`** and list all target languages to process.  
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

## Step 7: Generate Spectrograms

> Convert prosody-only audio into spectrograms for model training.

1. In **`config.py`**, set **`AUDIO_SAVED`** to the directory where you want to store spectrograms.  
2. Open **`scripts/make_spect.py`** and ensure the `location` variable matches the `save_directory`.  
3. Optionally, adjust parameters such as:
   - `n_fft`
   - `hop_length`
   - `sr`
   - the spectrogram function itself  
4. Run the script:

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
