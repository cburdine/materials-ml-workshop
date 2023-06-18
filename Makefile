build:
	jupyter-book build MaterialsML/

fullbuild:
	jupyter-book build --all MaterialsML/

release: build
	ghp-import -n -p -f MaterialsML/_build/html
