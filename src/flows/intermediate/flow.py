"""Flow for preparing the raw dataset."""

from random import random

from metaflow import FlowSpec, Parameter, current, step
from utils.config import get_dataset_path
from utils.logging import bprint


class IntermediateFlow(FlowSpec):
    """
    IntermediateFlow class represents a Metaflow flow for processing
    the raw Spotify dataset.

    Attributes
    ----------
    raw_dir : Path
        The directory path for the raw data.
    processed_dir : Path
        The directory path for the processed data.
    make_smaller : Parameter
        A parameter to control whether to create a smaller
        dataset for testing purposes.

    Methods
    -------
    start()
        Read the raw data, clean up the column names, add a
        row id, and dump to parquet.
    end()
        End the flow.
    """

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-instance-attributes

    raw_spotify_dir = get_dataset_path('raw_spotify')
    master_spotify_dir = get_dataset_path('master_spotify')

    subset = Parameter(
        'subset',
        default=False,
        type=bool,
        help='Create a smaller dataset for testing purposes',
    )

    @step
    def start(self):
        """Start the flow and print a cool message."""
        bprint("🌀 Let's get started")
        bprint(f'Running: {current.flow_name} @ {current.run_id}')
        bprint(f'User: {current.username}')
        self.next(self.clean_data)

    @step
    def clean_data(self):
        """Read the raw data, clean it and dump to parquet."""
        import pandas as pd

        def select_random(i) -> bool:  # numpydoc ignore=PR01,RT01
            """Select a random row if subset is True."""
            return i > 0 and random() > 0.50

        bprint('Reading data', level=1)
        df_playlist = pd.read_csv(
            self.raw_spotify_dir,
            on_bad_lines='skip',
            skiprows=select_random if self.subset else None,
        )
        bprint(f'Total rows read: {len(df_playlist):,}', level=2)

        bprint('Cleaning up column names', level=1)
        df_playlist.columns = df_playlist.columns.str.replace('"', '')
        df_playlist.columns = df_playlist.columns.str.replace('name', '')
        df_playlist.columns = df_playlist.columns.str.replace(' ', '')

        bprint('Adding a row id', level=1)
        df_playlist.insert(0, 'row_id', range(0, len(df_playlist)))

        bprint('Dumping to parquet', level=1)
        df_playlist.to_parquet(self.master_spotify_dir)
        bprint(f'Total rows: {len(df_playlist):,}', level=2)
        bprint(f'Saved at {self.master_spotify_dir}', level=2)
        self.next(self.end)

    @step
    def end(self):
        """End the flow adn print a cool message."""
        bprint('✨ All done ᕙ(⇀‸↼‶)ᕗ')


if __name__ == '__main__':
    IntermediateFlow()
