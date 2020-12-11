# Website Fingerprinting Tools

This python libraries defines a set of tools for website fingerprinting research.
It includes code for fetching and recording website traces as well as feature extraction and classification.


| Component                                               | Description                                                                                                 |
|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [**lab.classifiers**](lab/classifiers/)                 | The [Deep Fingerprinting][1], [k-Fingerprinting][2], [p1-FP][3], and [Var-CNN][4] classifiers.              |
| [**lab.fetch_websites**](lab/fetch_websites.py)         | Selenium based orchestration for capturing and recording web-page loads.                                    |
| [**lab.sniffer**](lab/sniffer.py)                       | Asynchronous packet sniffers based on tcpdump and scapy.                                                    |
| [**lab.third_party.li2018measuring**](lab/third_party/) | Features from the paper "[Measuring Information Leakage in Website Fingerprinting Attacks and Defenses][6]" |
| [**lab.metrics**](lab/metrics.py)                       | Additional website-fingerprinting metrics such as [Wang's r-precision][5].                                  |


## Installation

This project is not currently listed on pypi.
Install it directly from the git repository using pip as follows.

```bash
python3 -m pip install git+https://github.com/jpcsmith/wf-tools.git
```

## Licence

This library has an MIT license, as found in the [LICENCE](./LICENCE) file.
The code in `lab/thirdparty/li2018measuring` is licensed under CRAPL v0 Beta 1.


[1]: https://doi.org/10.1145/3243734.3243768 (Deep Fingerprinting)
[2]: https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/hayes (k-Fingerprinting)
[3]: https://doi.org/10.2478/popets-2019-0043 (p1-FP)
[4]: https://doi.org/10.2478/popets-2019-0070 (Var-CNN)
[5]: https://doi.org/10.1109/SP40000.2020.00015 (r-precision)
[6]: https://doi.org/10.1145/3243734.3243832
