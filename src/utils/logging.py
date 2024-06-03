"""This module contains utility functions for logging."""

PREFIX_MAP = {
    0: '',
    1: '|',
}
for i, lvl in enumerate(range(2, 10), 1):
    PREFIX_MAP[lvl] = '|' + '   ' * i


def bprint(*args, level: int = 0, prefix: str = '', **kwargs) -> None:
    """
    A better print function that prints a message with a
    specified indentation level.

    Parameters
    ----------
    *args : tuple
        The message parts to be printed.
    level : int, optional
        The indentation level of the message. The default is 0.
    prefix : str, optional
        The user-defined prefix to be printed before the message. The default is ''.
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
    full_prefix = PREFIX_MAP[level]
    if prefix:
        full_prefix = ''.join([full_prefix[: -len(prefix)], prefix])
    print(full_prefix, *args, **kwargs)
