from distutils.core import setup
from os import mkdir, name
from shutil import copy

try: mkdir('scripts')
except FileExistsError: pass

script = 'scripts/jelly' + '.py' * (name == 'nt')
copy('jelly/__main__.py', script)

setup(
	name = 'jellylanguage',
	version = '0.1.5',
	packages = [
		'jelly'
	],
	scripts = [
		script
	],
	requires = [
		'sympy'
	]
)
