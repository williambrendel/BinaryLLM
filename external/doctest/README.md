# doctest

This directory needs `doctest.h` from the doctest project.

Download it once:

```bash
curl -L https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h -o external/doctest/doctest.h
```

Or via Homebrew:

```bash
brew install doctest
# then copy from $(brew --prefix doctest)/include/doctest/doctest.h
```

The file is a single header, ~9000 lines, header-only test framework.
