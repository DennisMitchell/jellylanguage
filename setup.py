from distutils.core import setup

setup(
	name = 'jellylanguage',
	version = '0.1.11',
	packages = [
		'jelly'
	],
	scripts = [
		'scripts/jelly'
	],
	requires = [
		'sympy'
	]
)
