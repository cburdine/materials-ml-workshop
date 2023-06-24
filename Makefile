build:
	jupyter-book build MaterialsML/

check:
	jupyter-book build --builder=custom --custom-builder spelling ./MaterialsML/

fullbuild:
	jupyter-book build --all MaterialsML/

release: build
	ghp-import -n -p -f MaterialsML/_build/html
