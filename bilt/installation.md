# Installation

See **[docs/installation.md](docs/installation.md)** for the full installation guide.

## Quick install

```bash
git clone https://github.com/Rikiza89/bilt.git
cd bilt
pip install -e .
```

## Verify

```python
import bilt
print(bilt.__version__)   # 0.2.0

from bilt import BILT
BILT.variants()
```
