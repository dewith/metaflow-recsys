# 🎧 Music Recommendation System

This repository contains code following the [Recommender Systems with Metaflow](https://outerbounds.com/docs/recsys-tutorial-overview/) tutorial[^1].


## Motivation

> **Can we suggest what to listen to next after a given song?**

Here we learn how to use DuckDB, Gensim, Metaflow, and Keras to build an end-to-end recommender system. The model learns from existing sequences (playlists by real users) how to continue extending an arbitrary new list. More generally, this task is also known as next event prediction (NEP). The modeling technique only leverage behavioral data in the form of interactions created by users when composing their playlists.


## Running instructions

### Environment

First, clone the repository and install all dependencies.

```bash
git clone https://github.com/dewith/music-recsys.git
cd music-recsys
make install
make pre-commit
source .env/bin/activate
```

### Dataset

Download the raw data from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists?resource=download).

You should put the csv file in `data/01_raw/spotify_dataset.csv`. Otherwise, you can change the raw filepath in `conf/base/catalog.yml`.

### Cloud

Now, configure Metaflow with AWS.

But before that, you just need to configure the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html) with your account.[^2].

```bash
metaflow configure aws
```

From there, you can just follow the Metaflow CLI instructions. Make sure that the S3 bucket you define (*actually*) exists in your AWS account.

### Execution

Finally, we can run the flows in the Metaflow project.

```bash
python src/flows/intermediate/flow.py run --subset 1
python src/flows/primary/flow.py run --dev 1 --ratio 0.1
python src/flows/modeling/flow.py run --knn_n 100
python src/flows/deployment/flow.py run --sagemaker_deploy 1
```

Here I wrote the default parameters for each flow, but they are all optional since they have default values. For the **Deployment Flow**, the default SageMaker image, instance and role are set in the `conf/base/params.yml`.

## Project overview
### Data

Music is ubiquitous in today's world-almost everyone enjoys listening to music. With the rise of streaming platforms, the amount of music available has substantially increased. While users may seemingly benefit from this plethora of available music, at the same time, it has increasingly made it harder for users to explore new music and find songs they like. Personalized access to music libraries and music recommender systems aim to help users discover and retrieve music they like and enjoy.

The used dataset is based on the subset of users in the #nowplaying dataset who publish their #nowplaying tweets via Spotify. In principle, the dataset holds users, their playlists and the tracks contained in these playlists.


#### Analysis
The following plots show the distribution of artists and songs in the final dataset.

![image](./data/06_viz/artists_songs_histogram.png)

Unsurprisingly, the majority of artists have few or no songs in users playlists and just a handful of the them appear more than 10k times in the dataset's playlists.

Given this behavior, we can use the [`powerlaw`](https://github.com/jeffalstott/powerlaw) package to compare the distribution of how artists are represented in playlists to a power law density function.

![image](./data/06_viz/artists_powerlaw.png)

### Model

The skip-gram model we trained is an embedding space: if we did our job correctly, the space is such that tracks closer in the space are actually similar, and tracks that are far apart are pretty unrelated.

This is a very powerful property, as it allows us to use the space to find similar tracks to a given one, or to find tracks that are similar to a given playlist.

A simple heuristic is to use the TSNE algorithm to visualize the latent space of the model and compare the closeness of tracks given their genre.

![image](./data/06_viz/tsne_latent_space.png)

While not perfect, we can see that rock and rap songs tend to be closer to each other in the latent space.

### Pipeline

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


[^1]: I've made some changes to the original tutorial to make it more readable, organized and/or robust.
[^2]: If not already configured, of course.
