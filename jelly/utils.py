class attrdict(dict):
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)
		self.__dict__ = self

class LazyImport:
	def __init__(self, *args):
		self.args = args
	def __getattr__(self, attr):
		self.__dict__ = __import__(*self.args).__dict__
		return self.__dict__[attr]

def lazy_import(names):
	names = names.split()
	if len(names) == 1:
		return LazyImport(*names)
	return map(LazyImport, names)
