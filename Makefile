.PHONY: docs

docs:
@sphinx-build -b html docs/source docs/_build/html
