# Contributing to Pix2Pix-Zero-LoRA

Thank you for your interest in contributing to this project! We welcome contributions from the community to improve the codebase, documentation, and research findings.

## Code of Conduct

Please be respectful and inclusive in all your interactions within this project.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please open an issue on GitHub. Include:

- A clear description of the issue.
- Steps to reproduce the bug.
- Your environment details (OS, Python version, GPU).

### 2. Suggesting Enhancements

We are always looking for ways to improve the pipeline. If you have an idea:

- Check if the feature has already been suggested.
- Open an issue to discuss the enhancement before starting work.

### 3. Pull Requests

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Ensure your code follows the project's style (we use `black` for formatting).
5. Add tests if applicable.
6. Submit a pull request with a clear description of your changes.

## Development Setup

To set up your environment for development:

```bash
git clone https://github.com/your-username/pix2pix-zero.git
cd pix2pix-zero
.\setup_windows.ps1
.\venv\Scripts\activate
pip install -e .
```

## Testing

We use `pytest` for testing. Run the tests with:

```bash
pytest
```

---

_Thank you for making this project better!_
