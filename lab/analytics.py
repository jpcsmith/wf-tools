"""Methods useful for analysis of results."""
import itertools
import pandas as pd


def median_difference(frame: pd.DataFrame, level: str = "") -> pd.DataFrame:
    """Return the differences of the medians of each column.
    """
    result_columns = {}
    medians = frame.groupby(level).median()
    level_values = medians.index.get_level_values(level)

    for lhs, rhs in itertools.combinations(level_values, 2):
        result_columns[f"({lhs} - {rhs})"] = medians.loc[lhs] - medians.loc[rhs]
    return pd.DataFrame(result_columns).T
