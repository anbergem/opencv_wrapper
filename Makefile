clean:
	rm -rf opencv-wrapper.egg-info dist build __pycache__

uninstall:
	pip uninstall opencv-wrapper

wheel:
	make clean
	python setup.py sdist bdist_wheel

upload-pypi:
	make clean
	make wheel
	twine upload dist/opencv-wrapper*

upload-test:
	make clean
	make wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*