"""Metaflow utility functions."""

from typing import Optional

from metaflow import Flow
from metaflow.client.core import Run


def get_latest_successful_run(flow_name: str) -> Optional[Run]:
    """
    Get the latest successful run for a given flow.

    Parameters
    ----------
    flow_name : str
        The name of the flow.

    Returns
    -------
    Run or None
        The latest successful run for the given flow, or
        None if no successful run is found.
    """
    for run in Flow(flow_name).runs():
        if run.successful:
            return run
    return None
