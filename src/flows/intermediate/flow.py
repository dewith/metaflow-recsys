"""Flow for preparing the raw dataset."""

import random
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, step
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

    raw_dir = Path('../data/01_raw/')
    processed_dir = Path('../data/02_processed/')

    make_smaller = Parameter(
        'make_smaller',
        default=False,
        type=bool,
        help='Create a smaller dataset for testing purposes',
    )

    @step
    def start(self):
        """Read the raw data, clean it and dump to parquet."""
        bprint("🌀 Let's get started")

        def select_random(i):  # numpydoc ignore=PR01,RT01
            """Select a random row if make_smaller is True."""
            return i > 0 and random.random() > 0.50

        bprint('Reading data', level=1)
        df_playlist = pd.read_csv(
            self.raw_dir / 'spotify_dataset.csv',
            on_bad_lines='skip',
            skiprows=select_random if self.make_smaller else None,
        )
        bprint(f'Total rows read: {len(df_playlist):,}', level=2)

        bprint('Cleaning up column names', level=1)
        df_playlist.columns = df_playlist.columns.str.replace('"', '')
        df_playlist.columns = df_playlist.columns.str.replace('name', '')
        df_playlist.columns = df_playlist.columns.str.replace(' ', '')

        bprint('Adding a row id', level=1)
        df_playlist.insert(0, 'row_id', range(0, len(df_playlist)))

        bprint('Dumping to parquet', level=1)
        filename = self.processed_dir / 'spotify_master.parquet'
        df_playlist.to_parquet(self.processed_dir / 'spotify_master.parquet')
        bprint(f'Total rows: {len(df_playlist):,}', level=2)
        bprint(f'Saved at {filename.resolve()}', level=2)
        self.next(self.end)

    @step
    def end(self):
        """End the flow adn print a cool message."""
        bprint('✨ All done ᕙ(⇀‸↼‶)ᕗ')


if __name__ == '__main__':
    IntermediateFlow()
