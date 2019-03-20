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