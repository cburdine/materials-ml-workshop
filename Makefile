build:
	jupyter-book build MaterialsML/

release: build
	ghp-import -n -p -f MaterialsML/_build/html
