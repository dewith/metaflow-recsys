"""Flow for preparing the proccessed dataset to be used to train the model."""

from random import choice

from gensim.models import KeyedVectors
from metaflow import FlowSpec, Parameter, card, current, step
from metaflow.cards import Markdown, Table
from utils.config import get_dataset_path, get_parameters
from utils.logging import bprint


class ModelingFlow(FlowSpec):
    """
    The flow for modeling and evaluating a recommendation system. This flow fit a
    Word2Vec model on the dataset, and evaluates the model on the validation dataset.

    Attributes
    ----------
    train_dir : str
        The path to save the train dataset.
    validation_dir : str
        The path to save the validation dataset.
    test_dir : str
        The path to save the test dataset.
    trained_model_dir : str
        The path to save the trained model.
    hyperparameters : dict
        The hyperparameters for the model.
    KNN_K : int
        The number of neighbors we retrieve from the vector space.

    Methods
    -------
    predict_next_track()
        Given an embedding space, predict best next song with KNN.
    evaluate_model()
        Evaluate the model on a given dataset of sequences.
    start()
        The starting step of the flow.
    train()
        Train multiple track2vec model on the train dataset.
    keep_best()
        Choose the best model based on the hit ratio.
    test()
        Evaluate the model on the test dataset.
    end()
        End the flow and print a cool message.
    """

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-instance-attributes

    # Datasets
    train_dir = get_dataset_path('train')
    validation_dir = get_dataset_path('validation')
    test_dir = get_dataset_path('test')
    trained_model_dir = get_dataset_path('track2vec_model')

    # Parameters
    hypers_grid = get_parameters()['hyperparameters_grid']
    KNN_K = Parameter(
        name='knn_n',
        help='Number of neighbors we retrieve from the vector space.',
        default=100,
        type=int,
    )

    def predict_next_track(
        self, vector_space: KeyedVectors, input_sequence: list, k: int
    ) -> list:
        """
        Given an embedding space, predict best next song with KNN.
        Initially, we just take the LAST item in the input playlist as the query
        item for KNN and retrieve the top K nearest vectors (you could think of
        taking the smoothed average embedding of the input list as a refinement).

        If the query item is not in the vector space, we make a random bet. We could
        refine this by taking for example the vector of the artist (average of all
        songs), or with some other strategy (sampling by popularity).

        For more options on how to generate vectors for "cold items" see the paper:
        https://dl.acm.org/doi/10.1145/3383313.3411477

        Parameters
        ----------
        vector_space : gensim.models.KeyedVectors
            The embedding space where tracks are represented as vectors.
        input_sequence : list
            The sequence of tracks (playlist) up to the current point.
        k : int
            The number of nearest neighbors to retrieve.

        Returns
        -------
        list
            A list of the top `k` similar tracks.
        """
        query_item = input_sequence[-1]
        if query_item not in vector_space:
            query_item = choice(list(vector_space.index_to_key))

        return [_[0] for _ in vector_space.most_similar(query_item, topn=k)]

    def evaluate_model(self, df, vector_space, k) -> float:
        """
        Evaluate the model's performance.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing test data with sequences of tracks.
        vector_space : gensim.models.KeyedVectors
            The embedding space where tracks are represented as vectors.
        k : int
            The number of nearest neighbors to retrieve for predictions.

        Returns
        -------
        float
            The hit rate, which is a measure of the model's accuracy in
            predicting the next track.
        """

        def predict(row):  # numpydoc ignore=PR01,RT01
            """Predict next tracks based on the vector space."""
            return self.predict_next_track(vector_space, row['track_test_x'], k)

        def did_hit(row):  # numpydoc ignore=PR01,RT01
            """Check if the predicted track is in the test set."""
            return 1 if row['track_test_y'] in row['predictions'] else 0

        df['predictions'] = df.apply(predict, axis=1)
        df['hit'] = df.apply(did_hit, axis=1)
        hit_rate = df['hit'].sum() / len(df)
        return hit_rate

    @step
    def start(self):
        """Start the flow and print a cool message."""
        bprint("🌀 Let's get started")
        bprint(f'Running: {current.flow_name} @ {current.run_id}')
        bprint(f'User: {current.username}')

        # We train K models in parallel, depending how many configurations of hypers
        # we set - we generate K set of vectors, and evaluate them on the validation
        # set to pick the best combination of parameters!
        self.next(self.train, foreach='hypers_grid')

    @step
    def train(self):
        """
        Generate vector representations for songs, based on the Prod2Vec idea.
        For an overview of the algorithm, see: https://arxiv.org/abs/2007.14906 .
        """
        import pandas as pd
        from gensim.models import Word2Vec

        bprint('Training word2vec model', level=1)
        bprint('Hyperparameters:', level=2)
        self.hypers = self.input
        for k, v in self.hypers.items():
            bprint(f'{k}: {v}', level=3)
        self.df_train = pd.read_parquet(self.train_dir)
        sequences = self.df_train['track_sequence'].apply(lambda x: x.tolist()).tolist()
        self.model = Word2Vec(sequences, **self.hypers)
        bprint('Training is completed!', level=2)
        bprint(f'Vector space size: {len(self.model.wv.index_to_key)}', level=3)

        bprint('Taking a glance', level=1)
        test_track = choice(list(self.model.wv.index_to_key))
        test_vector = self.model.wv[test_track]
        test_sims = self.model.wv.most_similar(test_track, topn=3)
        bprint(f"Test track: '{test_track}'", level=2)
        bprint(f'Vector [:5]: {test_vector[:5]}', level=2)
        bprint('Similar songs:', level=2)
        for song, distance in test_sims:
            bprint(f'{song} ({distance:.4f})', level=3)

        bprint('Evaluating with validation set', level=1)
        self.df_validate = pd.read_parquet(self.validation_dir)
        self.validation_metric = self.evaluate_model(
            df=self.df_validate, vector_space=self.model.wv, k=self.KNN_K
        )
        bprint(f'Hit ratio @ {self.KNN_K} is {self.validation_metric:.4f}', level=2)
        self.track_vectors = self.model.wv
        self.next(self.keep_best)

    @card(type='blank', id='grid_search_card')
    @step
    def keep_best(self, inputs):
        """
        Choose the best model based on the hit ratio.

        This method takes a list of input runs and selects the best model based on
        the hit ratio. It sorts the runs in descending order of hit ratio and
        prints the top runs along with their hyperparameters.

        It also generates a table of results showing the hyperparameters
        and hit ratio for each run.

        Parameters
        ----------
        inputs : list
            A list of input runs.
        """
        bprint('Choosing the best model', level=1)
        self.runs = []
        for input_run in inputs:
            self.runs.append(
                {
                    'model': input_run.model,
                    'vectors': input_run.track_vectors,
                    'hyperparameters': input_run.hypers,
                    'hit_ratio': input_run.validation_metric,
                }
            )

        # Pretty print the results
        self.runs = sorted(self.runs, key=lambda x: x['hit_ratio'], reverse=True)
        bprint('Top runs:', level=1)
        for i, run in enumerate(self.runs):
            bprint(f'Run {i+1} (Best!)' if i == 0 else f'Run {i+1}', level=2)
            bprint(f'Hit ratio @ {self.KNN_K} is {run["hit_ratio"]:.4f}', level=3)
            bprint('Hyperparameters:', level=3)
            for k, v in run['hyperparameters'].items():
                bprint(f'{k}: {v}', level=4)
            if i == 2:
                break

        # Create table of results in card
        hypers_names = [run['hyperparameters'].keys() for run in self.runs]
        hypers_names = list({item for sub in hypers_names for item in sub})
        headers = [*sorted(hypers_names), f'hit_ratio@{self.KNN_K}']
        table_data = []
        for run in self.runs:
            row = [run['hyperparameters'][k] for k in headers[:-1]]
            row.append(round(100 * float(run['hit_ratio']), 2))
            table_data.append(row)
        current.card.append(Markdown('# Results from parallel training'))
        current.card.append(Table(headers=headers, data=table_data))

        self.final_model = self.runs[0]['model']
        self.final_vectors = self.runs[0]['vectors']
        self.final_dataset = inputs[0].df_train

        self.final_model.save(str(self.trained_model_dir))
        bprint(f'Saved the best model at {self.trained_model_dir}', level=2)

        self.next(self.eval)

    @step
    def eval(self):
        """
        Test the generalization abilities of the best model by running
        predictions on the unseen test data.

        We report a quantitative point-wise metric, hit ratio @ K, as an
        initial implementation. However, evaluating recommender systems is a very
        complex task, and better metrics, through good abstractions, are available,
        i.e. https://reclist.io/ .
        """
        import pandas as pd

        bprint('Evaluating with test set', level=1)
        self.df_test = pd.read_parquet(self.test_dir)
        self.test_metric = self.evaluate_model(
            df=self.df_test, vector_space=self.final_vectors, k=self.KNN_K
        )
        bprint(f'Hit ratio @ {self.KNN_K} is {self.test_metric:.4f}', level=2)
        self.next(self.end)

    @step
    def end(self):
        """End the flow and print a cool message."""
        bprint('✨ All done ᕙ(⇀‸↼‶)ᕗ')


if __name__ == '__main__':
    ModelingFlow()
