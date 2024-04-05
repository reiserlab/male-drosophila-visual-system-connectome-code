""" Helper functions """

import re
import unicodedata


def slugify(value, to_lower=True, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    original source: https://docs.djangoproject.com/en/4.1/_modules/django/utils/text/#slugify
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[\(\)]", "_", value)
    value = re.sub(r"[^\w\s-]", "", value)
    if to_lower:
        value = value.lower()
    return re.sub(r"[-\s]+", "-", value)


def num_expand(
    my_str
  , to_lower:bool=True
  , expand:int=10
) -> str:
    """
    Expands the numeric parts of alphanumeric parts. This is useful for (more) natural sorting.

    For example, with an `expand` values of 3 the string 'aMe6a' becomes 'ame006a' and 'aMe10' 
    becomes 'ame010'. If the 'to_lower' is set to False, the return values would be 'aMe006a'.

    Parameters
    ----------
    my_str : str
        A string
    to_lower : bool, default=True
        convert string to lowercase as well. Useful for sorting.
    expand : int, default=10
        expands each set of numbers in a string to 10 digits.

    Returns
    -------
    num_expand : str
        my_str with the numbers expanded to 10 digits.

    Example
    -------
    
    Normal sorting of a list of strings can have this result:

    >>> sorted(['aMe10', 'aMe8', 'aMe6b'])
    ['aMe10', 'aMe6b', 'aMe8']

    Using `num_expand` to solve this:
    >>> sorted(['aMe10', 'aMe8', 'aMe6b'], key=num_expand)
    ['aMe6b', 'aMe8', 'aMe10']

    Sorting a pandas DataFrame could work like this:

    >>> import pandas as pd
    df = pd.DataFrame({'type':['aMe10', 'aMe8', 'aMe6b']})
    >>> df['tmp_sort'] = df['type'].apply(num_expand)
    >>> df.sort_values(by='tmp_sort')['type']
    2    aMe6b
    1     aMe8
    0    aMe10
    Name: type, dtype: object
    """

    if to_lower:
        my_str = my_str.lower()
    return re.sub(r'(\d+)', lambda x: f'{x.group(1).zfill(expand)}', my_str)
