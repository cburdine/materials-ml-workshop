# Materials + ML Workshop

This is the repository for the Materials + ML Workshop self-paced tutorial content.

These tutorials use the Jupyter Book platform, and are intended to be exported as a static site. Hopefully we can find somewhere (e.g. Github Pages) to host this content once it is completed.

## Building the site:

First you must install the Jupyter Book dependencies inyour Python environment:

```
pip3 install --upgrade jupyter-book notebook
```

If you have GNU make installed on your system, you can build the static site by invoking the Makefile by running the command
```
make
```

inside this directory. if Make is not installed, you can run the command:

```
jupyter-book build MaterialsML/
```

(If we have a more complex build process in the future, this may be subject to change).

Once you build the site, you can view it locally by opening `MaterialsML/_build/html/index.html` with your web browser.
