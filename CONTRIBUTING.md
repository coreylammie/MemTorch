## Contributing

Please follow the "fork-and-pull" Git workflow:

1.  **Fork** the repo on GitHub.
2.  **Clone** the project to your own machine using `git clone --recursive`.
3.  **Enter Development Mode** using `python setup.py develop` in the cloned repository's directory.
4.  **Configure** `git pre-commit`, `black`, `isort`, and `clang-format` using `pip install pre-commit black isort && pre-commit install` and `apt install clang clang-format` (for linux) or `choco install llvm uncrustify cppcheck` (for windows).
5.  **Commit** changes to your own branch.
6.  **Push** your work back up to your fork.
7.  Submit a **Pull request** so that your changes can be reviewed.

_Be sure to merge the latest from 'upstream' before making a pull request_. This can be accomplished using `git rebase master`.
