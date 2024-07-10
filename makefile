.PHONY: lint

lint: # run formatting and linting
	isort cellpack_analysis
	black cellpack_analysis