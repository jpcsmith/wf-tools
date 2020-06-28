"""Tests for the filter_by_checkpoint function in lab.fetch_websites.
"""
from collections import Counter

from lab.fetch_websites import filter_by_checkpoint, Result


def test_empty():
    """It should return the full sequence for no checkpoint."""
    counter = {'Q043': 10, 'tcp': 20}
    assert filter_by_checkpoint(
        urls=['https://google.com', 'https://example.com'], checkpoint=[],
        counter=counter
    ) == {
        'https://google.com': counter.copy(),
        'https://example.com': counter.copy()
    }

    counter = {'Q043': 1, 'tcp': 1}
    assert filter_by_checkpoint(
        urls=['https://mail.com', 'https://pie.com'], checkpoint=[],
        counter=counter
    ) == {'https://mail.com': counter.copy(), 'https://pie.com': counter.copy()}


def make_result(**kwargs) -> Result:
    """Make a result with the provided keys and defaults for others.
    """
    defaults: Result = {
        'url': '', 'protocol': '', 'page_source': None, 'final_url': None,
        'status': 'success', 'http_trace': [], 'packets': b''
    }
    defaults.update(kwargs)  # type: ignore
    return defaults


def test_existing_result():
    """It should filter out any results that already exist."""
    urls = ['https://google.com', 'https://example.com']
    counter = Counter({'Q043': 10, 'tcp': 20})

    result = filter_by_checkpoint(urls=urls, counter=counter, checkpoint=[
        make_result(url='https://google.com', protocol='tcp', status='success'),
        make_result(url='https://example.com', protocol='Q043',
                    status='success')
    ])

    assert result == {'https://google.com': (counter - Counter(tcp=1)),
                      'https://example.com': (counter - Counter(Q043=1))}


def test_no_negative_returns():
    """It should not return negative values."""
    urls = ['https://google.com', 'https://example.com']
    counter = Counter({'Q043': 1, 'tcp': 1})

    result = filter_by_checkpoint(urls=urls, counter=counter, checkpoint=[
        make_result(url='https://google.com', protocol='tcp', status='success'),
        make_result(url='https://google.com', protocol='tcp', status='success')
    ])

    assert result == {'https://google.com': Counter(Q043=1),
                      'https://example.com': counter.copy()}


def test_no_empty_returns():
    """It should not return urls that have no more protocols."""
    urls = ['https://google.com', 'https://example.com']
    counter = Counter({'Q043': 1, 'tcp': 1})

    result = filter_by_checkpoint(urls=urls, counter=counter, checkpoint=[
        make_result(url='https://google.com', protocol='tcp', status='success'),
        make_result(url='https://google.com', protocol='Q043', status='success')
    ])

    assert result == {'https://example.com': counter.copy()}


def test_check_sequential_failures():
    """If a url failed n times sequentially in the checkpoint,
    it should not be returned.
    """
    urls = ['https://a.com', 'https://b.com']
    counter = Counter({'Q043': 1, 'tcp': 1})

    result = filter_by_checkpoint(urls=urls, counter=counter, checkpoint=[
        make_result(url='https://a.com', protocol='tcp', status='success'),
        make_result(url='https://a.com', protocol='Q043', status='failure'),
        make_result(url='https://a.com', protocol='Q043', status='timeout'),
        make_result(url='https://a.com', protocol='Q043', status='failure')
    ], max_attempts=3)

    assert result == {'https://b.com': counter.copy()}


def test_non_sequential_failures():
    """If a url failed n times non-sequentially in the checkpoint,
    it is okay.
    """
    urls = ['https://a.com', 'https://b.com']
    counter = Counter({'Q043': 3, 'tcp': 3})

    result = filter_by_checkpoint(urls=urls, counter=counter, checkpoint=[
        make_result(url='https://a.com', protocol='tcp', status='success'),
        make_result(url='https://a.com', protocol='Q043', status='failure'),
        make_result(url='https://a.com', protocol='Q043', status='success'),
        make_result(url='https://a.com', protocol='Q043', status='failure'),
        make_result(url='https://a.com', protocol='Q043', status='success'),
        make_result(url='https://a.com', protocol='Q043', status='failure')
    ], max_attempts=3)

    assert result == {'https://a.com': (counter - Counter(tcp=1, Q043=2)),
                      'https://b.com': counter.copy()}


def test_no_success_max_failures():
    """It should correctly handle items which have only ever failed."""
    checkpoint = [
        make_result(url="https://www.a.com", protocol="Q043", status="failure"),
        make_result(url="https://www.a.com", protocol="Q043", status="failure"),
        make_result(url="https://www.a.com", protocol="Q043", status="failure"),
    ]
    urls = ["https://www.a.com"]
    version_ctr: Counter = Counter(Q043=1, Q046=1)
    assert filter_by_checkpoint(urls, checkpoint, version_ctr) == Counter()
