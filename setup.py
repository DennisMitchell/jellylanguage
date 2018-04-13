from distutils.core import setup

setup(
	name = 'jellylanguage',
	version = '0.1.2',
	packages = [
		'jelly'
	],
	scripts = [
		'scripts/jelly'
	],
	requires = [
		'numpy',
		'sympy'
	]
)