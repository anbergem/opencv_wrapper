clean:
	rm -rf cvhelper.egg-info dist build __pycache__

uninstall:
	pip uninstall cvhelper

wheel:
	make clean
	python setup.py sdist bdist_wheel

upload-pypi:
	make clean
	make wheel
	twine upload dist/cvhelper*

upload-test:
	make clean
	make wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*