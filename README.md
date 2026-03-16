# Lyric NLP Theme Classifier

## Project Overview

A multi-label theme classification model for song lyrics built on Longformer. Given a song's lyrics the model identifies which of 12 themes are present and returns a confidence score for each. The project covers the full pipeline from :
+ Lyric scraping
+ Auto-labeling
+ Fine-tuning
+ Inference
  
This project fine-tunes `allenai/longformer-base-4096` on a custom dataset of 900+ song lyrics spanning a wide range of decades and genres. Because no labeled training data existed, a two-stage pipeline was built:

1. **Stage 1 — Auto-labeling:** `facebook/bart-large-mnli` is used as a zero-shot classifier to automatically assign theme labels to each song in the dataset. Long lyrics are chunked into 400-word segments to stay within the model's token limit, with scores averaged across chunks.

2. **Stage 2 — Fine-tuning:** The auto-labeled dataset is used to fine-tune Longformer for multi-label sequence classification. Longformer's sliding-window attention makes it well suited to longer lyrics that exceed standard BERT's 512 token limit.

The fine-tuned model and tokenizer are saved to [Hugging Face Hub](https://huggingface.co/gpecorino/lyric-theme-longformer).

## Themes

The model predicts the presence of these 12 themes:

| Theme ||
|---|---|
| Love and romance | Heartbreak and loss |
| Identity and self-discovery | Rebellion and resistance |
| Nostalgia and memory | Depression and mental health |
| Social commentary | Celebration and joy |
| Spirituality and faith | Ambition and success |
| Loneliness and isolation | Death and mortality |

These themes were created to capture as many popular music themes as possible without getting to granular. 

## About the Dataset

 **Size:** 900+ songs
- **Genre:** Mixed / various
- **Source:** Lyrics scraped from the [Genius API](https://genius.com/developers) using the `lyricsgenius` Python wrapper
- **Labels:** Auto-generated using zero-shot classification with `facebook/bart-large-mnli` — no manual labeling was performed
- **Format:** Stored locally as `lyrics_labeled.json` — not included in this repository due to copyright considerations
- 
The dataset is not publicly distributed. To reproduce it, follow the setup instructions below and run the scraping and labeling cells in the notebook using your own Genius API token.

The dataset for this project was created from scratch by me. I first generated a list of artists and songs spanning a wide range of genres with the help of generative AI that I stored in the [english_song_dataset.csv](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/english_song_dataset.csv). Then I used the [lyric_scraping notebook](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/lyric_scrapper.ipynb) to pull the lyrics using lyricgenius library and then process the lyrics to prepare them for use in the model. These lyrics were then stored in [lyric_list.txt](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/lyrics_list.txt) for easy access.



## Modeling

All the code for the model, including the auto-labeling, fine-tuning and inference are located in the [longform_bert_model notebook](https://github.com/gpecorino/Lyric_NLP_Analyzer/blob/main/longform_bert_model.ipynb). This notebook is broken up into 3 main parts:
1. Auto-Labeling
2. Fine-Tuning
3. Inference

### Stage 1


### Key design decisions

**Why Longformer?** Standard BERT models truncate at 512 tokens. Approximately 19% of songs in the dataset exceed 400 words, making Longformer's 4096 token limit a better fit for full lyric ingestion without truncation.

**Why zero-shot labeling?** No pre-labeled lyrics dataset existed for this theme taxonomy. `facebook/bart-large-mnli` scores each theme against the lyrics using Natural Language Inference, enabling automatic labeling of 900+ songs without manual annotation.

**Why multi-label?** Songs commonly express more than one theme simultaneously. Multi-label classification with sigmoid activation allows each theme to be predicted independently rather than forcing a single category per song.


## Areas for Improvement



## Technologies

+ Python: Primary language used for data collection, data processing and model training
+ Pandas: Used for data manipulation 
+ Pytorch:
+ Scikit-Learn:
+ Lyricgenius:
+ LongForm Bert Model:
