# Materials + ML Workshop

This is the repository for the Materials + ML Workshop self-paced tutorial content.

These tutorials use the Jupyter Book platform, and are intended to be exported as a static site. Hopefully we can find somewhere (e.g. Github Pages) to host this content once it is completed.

## Building the site:

### Dependencies:

First you must install the Jupyter Book dependencies inyour Python environment:

```
pip install --upgrade jupyter-book notebook
```

You will also need to install the dependencies used in the tutorials. These can be installed with:

```
pip install -r MaterialsML/requirements.txt
```

### Build HTML:

If you have GNU make installed on your system, you can build the static site by invoking the Makefile by running the command
```
make
```

inside this directory. if Make is not installed, you can run the command:

```
jupyter-book build MaterialsML/
```

Either of these commands will build the site using data cached from previous builds, if available. To completely rebuild the site from scratch, you can run `make fullbuild`.

Once you build the site, you can start a local web server by running the command:
```
python3 -m http.server --directory ./MaterialsML/_build/html/
```
This will start up a server you can access in your web browser at [http://localhost:8000/](http://localhost:8000).

Alternatively, you can view the HTML source files locally by opening `MaterialsML/_build/html/index.html` with your web browser.

### Checking Markdown Files

To check the spelling of markdown files, run the command: `make check`.

### Deploy to Github Pages

To deploy the main branch to the Github pages site, run the `make` command with the `release` target:

```
make release
```

*Warning*: This will push to the `gh-pages` branch and update the live site with the snapshot of the site built in the `MaterialsML/_build` directory. Be sure that the latest changes have been committed before deploying.
