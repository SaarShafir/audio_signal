# Installation

I recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:


## Install from GitHub

To install the latest version directly from GitHub:

```bash
pip install git+https://github.com/SaarShafir/audio_signal.git
```

Or with uv:

```bash
uv pip install git+https://github.com/SaarShafir/audio_signal.git
```

## Install from PyPI

To install the latest release from PyPI:

```bash
pip install audio-signal
```

Or with uv:

```bash
uv pip install audio-signal
```

## Development Setup

To set up a development environment:

```bash
git clone https://github.com/SaarShafir/audio_signal.git
cd audio_signal
pip install -e .[dev]
```

Or with uv:

```bash
uv pip install -e .[dev]
```

## Troubleshooting

- Ensure you are using Python 3.7 or newer.
- If you encounter permission errors, try adding `--user` to the pip command.
- For issues with `uv`, see the [uv documentation](https://github.com/astral-sh/uv).
- If you have problems installing dependencies, try upgrading pip:

  ```bash
  pip install --upgrade pip
  ```

For further help, open an issue on [GitHub](https://github.com/SaarShafir/audio_signal/issues).

