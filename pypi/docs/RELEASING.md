# Building and publishing CEJSP

Run these commands from the repository root. No token belongs in pyproject.toml. Build first; enter your PyPI token only when Twine requests it during upload. Publication is an explicit maintainer action.

```powershell
.\.venv\Scripts\python.exe -m build --no-isolation --outdir pypi/dist/0.5.0 pypi
.\.venv\Scripts\python.exe -m twine check --strict pypi/dist/0.5.0/cejsp-0.5.0-py3-none-any.whl pypi/dist/0.5.0/cejsp-0.5.0.tar.gz
```

Run the tests and installed-artifact integration checks before uploading. The source distribution includes only runtime sources, the retained reference, English user documentation, examples, package metadata, and licenses. Historical tests and research experiments remain in the repository for development, not in the release archive.

When ready to publish to the real PyPI:

```powershell
.\.venv\Scripts\python.exe -m twine upload pypi/dist/0.5.0/cejsp-0.5.0-py3-none-any.whl pypi/dist/0.5.0/cejsp-0.5.0.tar.gz
```

Use a token for the correct service. An existing release file cannot be replaced on PyPI; increment the version and rebuild if it has already been uploaded. These commands do not check ownership or name availability in advance. Never upload older files with a wildcard spanning multiple releases.
