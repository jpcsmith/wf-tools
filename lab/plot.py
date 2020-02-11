"""Functionality related to plotting and figures."""
from pathlib import Path
from typing import Optional, Sequence, Union

__all__ = ['savefig_with_formats']


def savefig_with_formats(figure, basename: Union[str, Sequence[str]],
                         formats: Optional[Sequence[str]] = None,
                         **save_kwargs):
    """Save the provided figure once for each extension format.

    If no formats are provided, save the figure as 'png' and 'pgf'.
    All other keyword arguments are passed to the savefig call.  By
    default, bbox_inches is set to 'tight' and dpi to 300.

    Alternately, basename can be a sequence of output files with the
    same root but different extensions. If this case, the figure is
    saved once per file.
    """
    if not isinstance(basename, str) and formats is not None:
        raise ValueError(f"Cannot have sequences of filenames and formats, "
                         f"for files {basename}, and formats {formats}")
    if not isinstance(basename, str):
        unique_bases = set(str(Path(name).with_suffix('')) for name in basename)
        if len(unique_bases) != 1:
            raise ValueError(f"Basenames should be the same: {unique_bases}")

    if isinstance(basename, str):
        if isinstance(formats, str):
            formats = [formats]
        else:
            formats = formats or ['png', 'pgf']
    else:  # It's a sequence
        assert formats is None
        formats = [Path(name).suffix for name in basename]
        assert '' not in formats
        formats = [fmt.lstrip('.') for fmt in formats]
        formats = ['pgf' if fmt == 'tex' else fmt for fmt in formats]
        basename = str(Path(basename[0]).with_suffix(''))

    save_kwargs.setdefault('dpi', 300)
    save_kwargs.setdefault('bbox_inches', 'tight')

    for fmt in formats:
        extension = fmt if fmt != 'pgf' else 'tex'
        figure.savefig(f'{basename}.{extension}', format=fmt, **save_kwargs)
