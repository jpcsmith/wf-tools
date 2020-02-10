"""Functionality related to plotting and figures."""
from typing import Optional, Sequence

__all__ = ['savefig_with_formats']


def savefig_with_formats(figure, basename: str,
                         formats: Optional[Sequence[str]] = None,
                         **save_kwargs):
    """Save the provided figure once for each extension format.

    If no formats are provided, save the figure as 'png' and 'pgf'.
    All other keyword arguments are passed to the savefig call.  By
    default, bbox_inches is set to 'tight' and dpi to 300.
    """
    if isinstance(formats, str):
        formats = [formats]
    else:
        formats = formats or ['png', 'pgf']

    save_kwargs.setdefault('dpi', 300)
    save_kwargs.setdefault('bbox_inches', 'tight')

    for fmt in formats:
        extension = fmt if fmt != 'pgf' else 'tex'
        figure.savefig(f'{basename}.{extension}', format=fmt, **save_kwargs)
