# Lyric NLP Theme Classifier

## Project Overview

A multi-label theme classification model for song lyrics built on [Longformer](https://huggingface.co/allenai/longformer-base-4096). Given a song's lyrics, the model identifies which of 12 themes are present and returns a confidence score for each. This project covers the full pipeline from :
+ Lyric scraping
+ Auto-labeling
+ Fine-tuning
+ Inference
  
This project fine-tunes `allenai/longformer-base-4096` on a custom dataset of 900+ song lyrics spanning a wide range of decades and genres. Because no labeled training data existed, a two-stage pipeline was built:

1. **Stage 1 — Auto-labeling:** `facebook/bart-large-mnli` is used as a zero-shot classifier to automatically assign theme labels to each song in the dataset. Long lyrics are chunked into 400-word segments to stay within the model's token limit, with scores averaged across chunks.

2. **Stage 2 — Fine-tuning:** The auto-labeled dataset is used to fine-tune Longformer for multi-label sequence classification. Longformer's sliding-window attention makes it well suited to longer lyrics that exceed standard BERT's 512 token limit.

The fine-tuned model and tokenizer are saved to [Hugging Face Hub](https://huggingface.co/gpecorino/lyric-theme-longformer).

## Themes

The model predicts the presence of 12 themes, selected to capture the most common subjects in popular music without becoming too granular:

| Themes ||
|---|---|
| Love and romance | Heartbreak and loss |
| Identity and self-discovery | Rebellion and resistance |
| Nostalgia and memory | Depression and mental health |
| Social commentary | Celebration and joy |
| Spirituality and faith | Ambition and success |
| Loneliness and isolation | Death and mortality |

## About the Dataset
- **Size:** 900+ songs
- **Genre:** Mixed / various
- **Decades:** Wide range
- **Source:** Lyrics scraped from the [Genius API](https://genius.com/developers) using the `lyricsgenius` Python wrapper
- **Labels:** Auto-generated using zero-shot classification with `facebook/bart-large-mnli` — no manual labeling was performed
- **Format:** Stored locally as `lyrics_labeled.json` — not included in this repository due to copyright considerations

The dataset was built from scratch in three steps:

1. A list of artists and songs spanning a wide range of genres and decades was generated with the help of generative AI and stored in [english_song_dataset.csv](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/english_song_dataset.csv)
2. Lyrics were scraped and cleaned using the [lyric_scraping notebook](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/lyric_scrapper.ipynb) and stored in [lyrics_list.txt](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/lyrics_list.txt)
3. Theme labels were automatically assigned in Stage 1 of the [modeling notebook](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/longform_bert_model.ipynb) and the data was stored in the [lyrics_labeled.json](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/lyrics_labeled.json) for use during fine-tuning training.

## Modeling

All modeling code — including auto-labeling, fine-tuning, and inference — is located in the [longform_bert_model notebook](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/longform_bert_model.ipynb). The notebook is organized into three parts:

1. **Auto-labeling** — zero-shot theme assignment using `facebook/bart-large-mnli`
2. **Fine-tuning** — Longformer training on the auto-labeled dataset
3. **Inference** — loading the saved model and predicting themes for new songs

### Inference example

```python
# Search for a song and predict its themes
lyrics = song_search("The Night We Met", "Lord Huron")
themes = predict_themes(lyrics, threshold=0.5)
print(themes)
# → {'heartbreak and loss': 0.91, 'nostalgia and memory': 0.84, 'loneliness and isolation': 0.72}
```

The `threshold` parameter controls how selective the model is. Lower values surface more themes, higher values return only the most confident predictions.

### Key design decisions

**Why Longformer?** Standard BERT models truncate at 512 tokens. Approximately 19% of songs in the dataset exceed 400 words, making Longformer's 4096 token limit a better fit for full lyric ingestion without truncation.

**Why zero-shot labeling?** No pre-labeled lyrics dataset existed for this theme taxonomy. `facebook/bart-large-mnli` scores each theme against the lyrics using Natural Language Inference, enabling automatic labeling of 900+ songs without manual annotation.

**Why multi-label?** Songs commonly express more than one theme simultaneously. Multi-label classification with sigmoid activation allows each theme to be predicted independently rather than forcing a single category per song.

### Performance
The model was trained for 6 epochs on 85% of the labeled dataset and evaluated against a held-out validation set of 15%. Both training loss and validation loss decreased consistently across all epochs, indicating the model learned without significant overfitting.

| Epoch | Training Loss | Validation Loss | F1 Micro | F1 Macro |
|---|---|---|---|---|
| 1 | 0.6489 | 0.4233 | 0.8777 | 0.7780 |
| 2 | 0.4083 | 0.3923 | 0.8975 | 0.8522 |
| 3 | 0.3642 | 0.3440 | 0.9062 | 0.8796 |
| 4 | 0.3235 | 0.3311 | 0.9120 | 0.8954 |
| 5 | 0.2972 | 0.3247 | **0.9130** | **0.8941** |
| 6 | 0.2852 | 0.3236 | 0.9115 | 0.8928 |

The best performing checkpoint was saved at **epoch 5** based on peak F1 Micro score.

#### Metrics explained

- **F1 Micro** measures overall performance across all theme predictions weighted by frequency, a score of 0.913 means the model is correctly identifying themes in the vast majority of cases across the full validation set
- **F1 Macro** averages performance equally across all 12 themes regardless of how often they appear, a score of 0.894 indicates the model performs consistently well even on less common themes, not just the most frequent ones

#### Notes on ROC AUC

ROC AUC returned `nan` during training. This occurs when a theme has no positive examples in the validation split, which can happen with rare themes in a 15% holdout of 900 songs. This does not reflect a problem with the model, the F1 scores are the more reliable performance indicator for this dataset size. Per-label ROC AUC can be computed after training using a larger evaluation set.

## Areas for Improvement
The model captures many themes present in song lyrics but several limiting factors were encountered during development, including hardware performance constraints. The following areas represent the most impactful opportunities for improvement:

1. **Increase the amount of training data**: The dataset of 900+ songs is functional but on the smaller side for fine-tuning a transformer model. Expanding to 2000+ songs, particularly for underrepresented themes, would likely improve generalization and reduce overfitting.

2. **Improve training label quality**: Theme labels were generated automatically using zero-shot classification, which introduces noise. Manually reviewing and correcting a subset of the lowest-confidence labels would directly improve the quality of the training signal the model learns from.

3. **Train for more epochs with a tuned learning rate schedule**: Training was limited to 6 epochs due to hardware and time constraints. Additional epochs combined with a cosine learning rate schedule could allow the model to converge more fully, particularly for less common themes.

4. **Per-label confidence thresholds**: The current inference uses a single threshold of 0.5 for all themes. Some themes score lower even when correctly detected. Tuning a separate threshold per label against the validation set would improve precision and recall across the board.

5. **Address class imbalance**: Some themes appear far more frequently in the dataset than others. Applying class weights during training would penalize errors on rare themes more heavily, encouraging the model to learn them rather than defaulting to the most common ones.


## Technologies

+ Python: Primary language used for data collection, data processing and model training
+ Pandas: Data loading and manipulation 
+ Pytorch: Tensor operations, model training, and GPU management
+ Scikit-Learn: Train/validation splitting and evaluation metrics (F1, ROC AUC)
+ Lyricgenius: Genius API wrapper for scraping song lyrics
+ Hugging Face Transformers: Longformer model architecture, tokenizer, and Trainer API
+ Hugging Face Hub: Cloud storage for fine-tuned model weights
