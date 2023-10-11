# Fine_Tune_Bert_for_Sentiment_Analisys

This repository contains a Python script for sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art deep learning model for natural language processing. The script is designed to fine-tune a pre-trained BERT model on a custom sentiment analysis task.

## Table of Contents

- [Overview](#overview)
- [Data Management](#data-management)
- [Training/Validation Split](#trainingvalidation-split)
- [Loading Tokenizer and Encoding Data](#loading-tokenizer-and-encoding-data)
- [Setting Up BERT Pretrained Model](#setting-up-bert-pretrained-model)
- [Creating Data Loaders](#creating-data-loaders)
- [Setting Up Optimizer and Scheduler](#setting-up-optimizer-and-scheduler)
- [Performance Metrics](#performance-metrics)
- [Training Loop for Fine-Tuning BERT](#training-loop-for-fine-tuning-bert)
- [Loading Fine-Tuned BERT Model and Evaluation](#loading-fine-tuned-bert-model-and-evaluation)

## Overview

Sentiment analysis, also known as opinion mining, is the process of determining the sentiment expressed in a piece of text, such as positive, negative, or neutral. This repository provides a Python script that leverages the power of BERT to perform sentiment analysis. BERT is known for its contextual understanding of text, making it well-suited for various NLP tasks, including sentiment analysis.

## Data Management

The script starts by loading and preparing a dataset named 'smile-annotations-final.csv,' which appears to contain text data with corresponding sentiment categories.

## Training/Validation Split

The dataset is divided into training and validation sets using scikit-learn's `train_test_split`. This split is crucial for assessing the model's performance.

## Loading Tokenizer and Encoding Data

Hugging Face Transformers library is used to load a BERT tokenizer. The script then encodes the text data, adding special tokens and creating attention masks.

## Setting Up BERT Pretrained Model

A pre-trained BERT model is initialized for sequence classification. The model can handle multiple labels, as indicated by `num_labels`.

## Creating Data Loaders

The script uses PyTorch data loaders to efficiently load data in mini-batches. Data is shuffled for randomness. Training and validation data loaders are created.

## Setting Up Optimizer and Scheduler

The optimizer used is AdamW, specifically designed for fine-tuning transformer models. A linear learning rate scheduler with warm-up is set up.

## Performance Metrics

Two custom functions are defined for performance evaluation:
   - `f1_score_func`: Calculates the F1 score, a key metric for model performance.
   - `accuracy_per_class`: Calculates the accuracy per class, offering insights into model performance across different categories.

## Training Loop for Fine-Tuning BERT

The main training loop fine-tunes the pre-trained BERT model using the training data. It runs for a specified number of epochs, computes training loss, and updates model parameters using backpropagation.

## Loading Fine-Tuned BERT Model and Evaluation

After training, the script loads the fine-tuned model and evaluates its performance using the validation data. It calculates the validation loss and the F1 score for weighted accuracy. Additionally, it prints the accuracy per class.

This repository is a valuable resource for developers and researchers interested in sentiment analysis using BERT. Feel free to adapt the code for your specific use case and datasets.

Happy coding!
