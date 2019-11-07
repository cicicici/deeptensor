.PHONY : install build wheel uninstall clean

PROJECT ?= deeptensor

build:
	python setup.py build

install-egg:
	python setup.py install

install: wheel
	pip install --force-reinstall dist/*.whl

wheel:
	python setup.py sdist bdist_wheel

uninstall: clean
	pip uninstall -y $(PROJECT)

clean:
	rm -rf build $(PROJECT).egg-info dist
