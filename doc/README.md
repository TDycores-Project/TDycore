# TDycore: Documentation

This page provides a brief description of the documentation for the TDycore
library.

## Quick build

If you have Python and Doxygen installed, these two commands should build the
documentation in `doc/sphinx/build/html/`.

```sh
pip install --user -r doc/sphinx/requirements.txt  # only needed once
make doc                                           # builds HTML site
```

## Sphinx

Sphinx is the tool used for TDycore's User Manual. Sphinx can produce
documentation in different output formats: HTML, LaTeX (for printable PDF
versions), ePub, Texinfo, manual pages, and plain text. Sphinx comes with a
broad set of extensions for different features:

* automatic inclusion of documentation from docstrings and code snippets
* support for TODO items
* syntax highlighting
* math rendering

To be able to contribute to TDycore's User Manual, you must install
[Sphinx](http://www.sphinx-doc.org/en/master/usage/installation.html) along with
any desired extensions.

The Sphinx API documentation depends on Doxygen's XML output (via the `breathe`
plugin). Build these files in the `xml/` directory via:

```sh
make doxygen
```

If you are editing documentation, such as the files in `doc/sphinx/source`, you
can rebuild incrementally via

```sh
make -C doc/sphinx html
```

This will generate HTML documentation in the [doc/sphinx/build](./sphinx/build)
directory. Use

```sh
make -C doc/sphinx latexpdf
```

to build a PDF using the LaTeX toolchain (which must be installed separately).
This requires the `rsvg-convert` utility, which is likely available from your
package manager under `librsvg` or `librsvg2-bin`.

For more Sphinx features, see

```sh
make -C doc/sphinx help
```

### Dependencies

Some of Sphinx's extensions require installation. They are distributed on
[PyPI](https://pypi.org) and can be installed with `pip`. The extensions used
for this project can be found in the [requirements file](./sphinx/requirements.txt)
and can be readily installed by running:

```sh
pip install --user -r doc/sphinx/requirements.txt
```

from the top level of the source tree.

You can use `virtualenv` to keep these extensions isolated, and to check that
all necessary extensions are included.

```sh
virtualenv VENV                              # create a virtual environment
. VENV/bin/active                            # activate the environment
pip install -r doc/sphinx/requirements.txt   # install dependencies inside VENV
make doc
```
