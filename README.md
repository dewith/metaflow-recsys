# 🎧 Playlist Recommendation System

This repository contains code following the [Recommender Systems with Metaflow](https://outerbounds.com/docs/recsys-tutorial-overview/) tutorial[^1].


## Motivation

> **Can we suggest what to listen to next after a given song?**

Here we learn how to use DuckDB, Gensim, Metaflow, and Keras to build an end-to-end recommender system. The model learns from existing sequences (playlists by real users) how to continue extending an arbitrary new list. More generally, this task is also known as next event prediction (NEP). The modeling technique only leverage behavioral data in the form of interactions created by users when composing their playlists.



[^1]: I've made some changes to the original tutorial to make it more readable, organized and/or robust.

## Data

Music is ubiquitous in today's world-almost everyone enjoys listening to music. With the rise of streaming platforms, the amount of music available has substantially increased. While users may seemingly benefit from this plethora of available music, at the same time, it has increasingly made it harder for users to explore new music and find songs they like. Personalized access to music libraries and music recommender systems aim to help users discover and retrieve music they like and enjoy.

The used dataset is based on the subset of users in the #nowplaying dataset who publish their #nowplaying tweets via Spotify. In principle, the dataset holds users, their playlists and the tracks contained in these playlists.

The dataset can be downloaded [here](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists?resource=download).

### Analysis
The following plots show the distribution of artists and songs in the final dataset.

![image](./data/06_viz/artists_songs_histogram.png)

Unsurprisingly, the majority of artists have few or no songs in users playlists and just a handful of the them appear more than 10k times in the dataset's playlists.

Given this behavior, we can use the [`powerlaw`](https://github.com/jeffalstott/powerlaw) package to compare the distribution of how artists are represented in playlists to a power law density function.

![image](./data/06_viz/artists_powerlaw.png)

## Model

The skip-gram model we trained is an embedding space: if we did our job correctly, the space is such that tracks closer in the space are actually similar, and tracks that are far apart are pretty unrelated.

This is a very powerful property, as it allows us to use the space to find similar tracks to a given one, or to find tracks that are similar to a given playlist.

A simple heuristic is to use the TSNE algorithm to visualize the latent space of the model and compare the closeness of tracks given their genre.

![image](./data/06_viz/tsne_latent_space.png)

While not perfect, we can see that rock and rap songs tend to be closer to each other in the latent space.

## Pipeline

The pipeline is composed of the following flows and steps:
1. Intermediate (`src/flows/intermediate/flow.py`)
    1. `clean_data`: Read the raw data, clean up the column names, add a row id, and dump to parquet.
 2. Primary (`src/flows/primary/flow.py`)
    1. `prepare_dataset`: Prepare the dataset by reading the parquet dataset and using DuckDB SQL-based wrangling.
 3. Modeling (`src/flows/modeling/flow.py`)
    1. `train`: Train multiple track2vec model on the train dataset using a hyperparameter grid.
    2. `keep_best`: Choose the best model based on the hit ratio.
    3. `eval`: Evaluate the model on the test dataset.
 4. Deployment (`src/flows/deployment/flow.py`)
    1. `build`: Take the embedding space, build a Keras KNN model and store it in S3.
    2. `deploy`: Construct a TensorFlowModel from the tar file in S3 and deploy it to a SageMaker endpoint.
    3. `check`: Check the SageMaker endpoint is working properly.

Visually, the pipeline looks like this:

![image](./data/06_viz/pipeline.png)

## Conclusion

In this little project we learned to:

- take a recommender system idea from prototype to real-time production;
- leverage Metaflow to train different versions of the same model and pick the best one;
- use Metaflow cards to save important details about model performance;
- package a representation of your data in a keras object that you can deploy directly from the flow to a cloud endpoint with AWS Sagemaker.
