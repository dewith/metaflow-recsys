"""Script that creates the styling utils."""

import pandas as pd
import seaborn as sns


def get_palette():
    """
    Get the color palette used for styling.

    Returns
    -------
    tuple
        A tuple containing the color palette and other styling colors.

        The color palette is a list of color codes used for different elements.
        The other styling colors include:
        - lines: The color code for lines.
        - text: The color code for regular text.
        - text_lighter: The color code for lighter text.
        - text_darker: The color code for darker text.
        - background: The color code for the background.
    """
    green = '#4c9878'
    orange = '#e87e7e'
    purple = '#b170d2'
    blue = '#366eff'
    palette = [green, orange, purple, blue]

    text_darker = '#282828'
    text = '#918c88'
    text_lighter = '#b2afac'
    lines = '#dad9d7'
    background = '#faf7f4'
    return palette, lines, text, text_lighter, text_darker, background


def apply_styling():
    """
    Apply styling to the pandas and seaborn libraries.

    Returns
    -------
    dict
        A dictionary containing the styling parameters:
        - 'palette': The color palette used for plotting.
        - 'lines': The color of the lines in the plots.
        - 'text': The color of the text in the plots.
        - 'text_darker': A darker shade of the text color.
        - 'text_lighter': A lighter shade of the text color.
        - 'background': The background color of the plots.
    """
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    pd.set_option('display.max_colwidth', 50)

    palette, lines, text, text_lighter, text_darker, background = get_palette()
    params = {
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'savefig.pad_inches': 0.3,
        'figure.facecolor': background,
        'axes.facecolor': background,
        'axes.edgecolor': lines,
        'axes.labelcolor': text,
        'axes.titlecolor': text_darker,
        'text.color': text,
        'xtick.color': text_lighter,
        'ytick.color': text_lighter,
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.6,
        'axes.grid': False,
    }
    sns.set_theme(context='notebook', style='white', palette=palette, rc=params)
    return {
        'palette': palette,
        'lines': lines,
        'text': text,
        'text_darker': text_darker,
        'text_lighter': text_lighter,
        'background': background,
    }
