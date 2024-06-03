"""Model for deployment in a sagemaker endpoint."""

import random
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from utils.logging import bprint


def build_retrieval_model(songs_ids, songs_embeddings):
    """
    Build a retrieval model for recommending songs based on their embeddings.

    Parameters
    ----------
    songs_ids : list
        A list of song IDs.
    songs_embeddings : numpy.ndarray
        A 2D array of song embeddings.

    Returns
    -------
    tuple
        A tuple containing the retrieval model and the song index.

    Notes
    -----
    The retrieval model is built using the given song IDs and embeddings. It consists of a
    StringLookup layer for mapping song IDs to indices, and an Embedding layer for
    representing the song embeddings. The song index is created using a BruteForce layer
    from TensorFlow Recommenders (tfrs) library.

    The retrieval model can be used to recommend songs based on their embeddings.

    Examples
    --------
    songs_ids = ['song1', 'song2', 'song3']
    songs_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    model, song_index = build_retrieval_model(songs_ids, songs_embeddings)
    """
    embedding_size = songs_embeddings[0].shape[0]
    bprint(f'Num of embeddings: {len(songs_embeddings):,}', level=3)
    bprint(f'Embeddings dimensions: {embedding_size}', level=3)

    bprint('Adding a vector for unknown items', level=3)
    unknown_vector = np.zeros((1, embedding_size))
    embedding_matrix = np.vstack([unknown_vector, songs_embeddings])
    bprint('First item:', embedding_matrix[0][0:5], level=4)
    bprint('Shape of the matrix:', embedding_matrix.shape, level=4)
    assert embedding_matrix[0][0] == 0.0

    bprint('Initializing layers and network', level=3)
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=songs_ids, mask_token=None)
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False,
    )
    embedding_layer.build((None,))

    model = tf.keras.Sequential([lookup_layer, embedding_layer])

    bprint('Creating retrieval model', level=3)
    brute_force = tfrs.layers.factorized_top_k.BruteForce(model)
    song_index = brute_force.index(candidates=songs_embeddings, identifiers=songs_ids)
    return model, song_index


class RetrievalModel:
    """
    Build a retrieval model using TF recommender abstraction by packaging
    the vector space in a Keras object.

    We can ship the artifact "as is" to a Sagemaker endpoint, and
    benefit from the PaaS abstraction and hardware acceleration.

    Parameters
    ----------
    songs_ids : numpy.ndarray
        An array of song IDs.
    songs_embeddings : numpy.ndarray
        A 2D array of song embeddings.
    """

    def __init__(self, songs_ids: np.ndarray, songs_embeddings: np.ndarray):
        """Initialize the retrieval model."""
        bprint('Building retrieval model', level=2)
        model, song_index = build_retrieval_model(songs_ids, songs_embeddings)
        self.model = model
        self.song_index = song_index
        self.songs_ids = songs_ids
        self.songs_embeddings = songs_embeddings

    def test(self):
        """Test the retrieval model."""
        bprint('Testing retrieval model', level=2)
        test_index = random.randint(0, 10)
        test_id = self.songs_ids[test_index]
        self.pprint_recommendations(test_id, k=5)
        test_id = r'Alabimbombao ヽ(≧◡≦)八(o^ ^o)ノ'  # Unknown!
        self.pprint_recommendations(test_id, k=5)

    def get_recommendations(self, song_id: str, k: int = 10) -> Tuple:
        """
        Get recommendations for a given song.

        Parameters
        ----------
        song_id : int
            The ID of the song for which recommendations are to be generated.
        k : int, optional
            The number of recommendations to be returned. Default is 10.

        Returns
        -------
        tuple
            A tuple containing the song vector, recommendation scores, and recommendation IDs.
            - song_vector : Tensorflow Tensor
                The vector representation of the input song.
            - rec_scores : Tensorflow Tensor
                The scores of the recommended songs.
            - rec_ids : Tensorflow Tensor
                The IDs of the recommended songs.
        """
        song_vector = self.model(np.array([song_id]))
        rec_scores, rec_ids = self.song_index(tf.constant([song_id]), k=k)
        return song_vector, rec_scores, rec_ids

    def pprint_recommendations(self, song_id, k=10):
        """
        Print recommendations for a given song.

        Parameters
        ----------
        song_id : int
            The ID of the song for which recommendations are to be generated.
        k : int, optional
            The number of recommendations to be returned. Default is 10.
        """
        song_vector, rec_scores, rec_ids = self.get_recommendations(song_id, k=k)
        bprint('Song ID:', song_id, level=3, prefix='* ')
        bprint('Song vector:', song_vector.numpy()[0][:5], level=3)
        bprint('Recommendations after track:', level=3)
        for rec_score, rec_id in zip(rec_scores[0].numpy(), rec_ids[0].numpy()):
            rec_id = str(rec_id, 'utf-8')
            if rec_id != song_id:
                bprint(f'{rec_score:.2f} - {rec_id}', level=4)

    def save(self, *args, **kwargs):
        """
        Save the retrieval model.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to the save function.
        **kwargs : dict
            Keyword arguments to be passed to the save function.
        """
        self.song_index.save(*args, **kwargs)
