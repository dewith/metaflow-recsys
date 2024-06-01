"""Flow for preparing the proccessed dataset to be used to train the model."""

from metaflow import FlowSpec, Parameter, current, step
from utils.config import get_dataset_path
from utils.logging import bprint


class PrimaryFlow(FlowSpec):
    """
    The primary flow for training a Recommender System.

    This flow prepares the dataset, performs data wrangling using DuckDB,
    splits the dataset into train, validation, and test sets, and saves the
    datasets locally.

    Attributes
    ----------
    master_spotify_dir : str
        The path to the master Spotify dataset.
    model_input_dir : str
        The path to save the prepared dataset for model input.
    train_dir : str
        The path to save the train dataset.
    validation_dir : str
        The path to save the validation dataset.
    test_dir : str
        The path to save the test dataset.

    Methods
    -------
    start()
        The starting step of the flow.
    prepare_dataset()
        Prepare the dataset by reading the parquet dataset and using DuckDB
        SQL-based wrangling.
    end()
        End the flow and print a cool message.
    """

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-instance-attributes

    master_spotify_dir = get_dataset_path('master_spotify')
    model_input_dir = get_dataset_path('model_input_full')
    train_dir = get_dataset_path('train')
    validation_dir = get_dataset_path('validation')
    test_dir = get_dataset_path('test')

    IS_DEV = Parameter(
        name='dev',
        help='Flag for dev development, with a smaller dataset',
        default=False,
        type=bool,
    )

    @step
    def start(self):
        """Start the flow and print a cool message."""
        bprint("🌀 Let's get started")
        bprint(f'Running: {current.flow_name} @ {current.run_id}')
        bprint(f'User: {current.username}')
        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self):
        """
        Get the data in the right shape by reading the parquet dataset
        and using DuckDB SQL-based wrangling to quickly prepare the datasets for
        training our Recommender System.
        """
        import duckdb
        import numpy as np

        bprint('Loading the dataset in DuckDB', level=1)
        con = duckdb.connect(database=':memory:')
        con.execute(f"""
            CREATE TABLE playlists AS
                SELECT
                    *,
                    CONCAT(user_id, '-', playlist) as playlist_id,
                    CONCAT(artist, '-', track) as track_id,
                FROM
                    '{self.master_spotify_dir}'
            ;
        """)

        bprint('Checking the dataset', level=1)
        columns = ['row_id', 'user_id', 'track_id', 'playlist_id', 'artist']
        for col in columns:
            con.execute(f'SELECT COUNT(DISTINCT({col})) FROM playlists;')
            bprint(f'No of unique {col}: {con.fetchone()[0]}', level=2)

        bprint('First row:', level=2)
        con.execute("PRAGMA table_info('playlists');")
        column_names = [col[1] for col in con.fetchall()]
        con.execute('SELECT * FROM playlists LIMIT 1;')
        first_row = con.fetchone()
        for col, val in zip(column_names, first_row):
            bprint(f'{col}: {val}', level=3)

        bprint('Preparing the final dataframe', level=1)
        sampling_cmd = ''
        if self.IS_DEV:
            bprint('Subsampling data to 10%, since dev mode is enabled', level=2)
            sampling_cmd = ' USING SAMPLE 10 PERCENT (bernoulli)'

        dataset_query = f"""
            SELECT
                *
            FROM (
                SELECT
                    playlist_id,
                    LIST(artist ORDER BY row_id ASC) as artist_sequence,
                    LIST(track_id ORDER BY row_id ASC) as track_sequence,
                    array_pop_back(LIST(track_id ORDER BY row_id ASC)) as track_test_x,
                    LIST(track_id ORDER BY row_id ASC)[-1] as track_test_y
                FROM
                    playlists
                GROUP BY playlist_id
                HAVING len(track_sequence) > 2
            )
            {sampling_cmd}
            ;
        """
        con.execute(dataset_query)
        df = con.fetch_df()
        bprint(f'No rows: {len(df)}', level=2)
        con.close()

        bprint('Splitting the dataset', level=1)
        train, val, test = np.split(
            df.sample(frac=1, random_state=42), [int(0.7 * len(df)), int(0.9 * len(df))]
        )
        self.df_dataset = df
        self.df_train = train
        self.df_val = val
        self.df_test = test
        bprint(f'No train rows: {len(self.df_train)}', level=2)
        bprint(f'No val rows: {len(self.df_val)}', level=2)
        bprint(f'No test rows: {len(self.df_test)}', level=2)

        bprint('Saving the datasets locally', level=1)
        self.df_dataset.to_parquet(self.model_input_dir)
        self.df_train.to_parquet(self.train_dir)
        self.df_val.to_parquet(self.validation_dir)
        self.df_test.to_parquet(self.test_dir)
        bprint(f'All tables saved at {self.model_input_dir.parent}', level=2)
        self.next(self.end)

    @step
    def end(self):
        """End the flow and print a cool message."""
        bprint('✨ All done ᕙ(⇀‸↼‶)ᕗ')


if __name__ == '__main__':
    PrimaryFlow()
