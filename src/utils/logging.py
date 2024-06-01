"""This module contains utility functions for logging."""


def bprint(*args, level: int = 0, **kwargs) -> None:
    """
    A better print function that prints a message with a
    specified indentation level.

    Parameters
    ----------
    *args : tuple
        The message parts to be printed.
    level : int, optional
        The indentation level of the message. The default is 0.
    **kwargs : dict
        Additional keyword arguments to be passed to the print function.

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> bprint("Hello, world!", level=1)
    | Hello, world!
    """
    prefix_map = {
        0: '',
        1: '|',
        2: '|   ',
        3: '|      ',
    }
    print(prefix_map[level], *args, **kwargs)
