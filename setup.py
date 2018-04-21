from distutils.core import setup

setup(
	name = 'jellylanguage',
	version = '0.1.4',
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
