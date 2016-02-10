import ast, cmath, functools, itertools, math, sympy, sys

inf = float('inf')
nan = float('nan')

class attrdict(dict):
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)
		self.__dict__ = self

def arities(links):
	return [link.arity for link in links]

def create_chain(chain, arity):
	return attrdict(
		arity = arity,
		call = lambda x = None, y = None: variadic_chain(chain, (x, y))
	)

def create_literal(string):
	return attrdict(
		arity = 0,
		call = lambda: eval(string, False)
	)

def copy(value):
	atoms['®'].call = lambda: value
	return value

def depth(link):
	if type(link) != list and type(link) != tuple:
		return 0
	if not link:
		return 1
	return 1 + max(map(depth, link))

def depth_match(link, larg, rarg = None):
	return not ((hasattr(link, 'ldepth') and link.ldepth != depth(larg)) or (hasattr(link, 'rdepth') and link.rdepth != depth(rarg)))

def conv_dyadic_integer(link, larg, rarg):
	try:
		iconv_larg = int(larg)
		try:
			iconv_rarg = int(rarg)
			return link(iconv_larg, iconv_rarg)
		except:
			return iconv_larg
	except:
		try:
			return int(rarg)
		except:
			return 0

def conv_monadic_integer(link, arg):
	try:
		return link(int(arg))
	except:
		return 0

def div(dividend, divisor, floor = False):
	if divisor == 0:
		return nan if dividend == 0 else inf
	if divisor == inf:
		return 0
	if floor or (type(dividend) == int and type(divisor) == int and not dividend % divisor):
		return int(dividend // divisor)
	return dividend / divisor

def dyadic_chain(chain, args):
	larg, rarg = args
	for link in chain:
		if link.arity < 0:
			link.arity = 2
	if chain and arities(chain[0:3]) == [2, 2, 2]:
		ret = dyadic_link(chain[0], args)
		chain = chain[1:]
	elif leading_constant(chain):
		ret = niladic_link(chain[0])
		chain = chain[1:]
	else:
		ret = larg
	while chain:
		if arities(chain[0:3]) == [2, 2, 0] and leading_constant(chain[2:]):
			ret = dyadic_link(chain[1], (dyadic_link(chain[0], (ret, rarg)), niladic_link(chain[2])))
			chain = chain[3:]
		elif arities(chain[0:2]) == [2, 2]:
			ret = dyadic_link(chain[0], (ret, dyadic_link(chain[1], args)))
			chain = chain[2:]
		elif arities(chain[0:2]) == [2, 0]:
			ret = dyadic_link(chain[0], (ret, niladic_link(chain[1])))
			chain = chain[2:]
		elif arities(chain[0:2]) == [0, 2]:
			ret = dyadic_link(chain[1], (niladic_link(chain[0]), ret))
			chain = chain[2:]
		elif chain[0].arity == 2:
			ret = dyadic_link(chain[0], (ret, rarg))
			chain = chain[1:]
		elif chain[0].arity == 1:
			ret = monadic_link(chain[0], ret)
			chain = chain[1:]
		else:
			print('Skipped atom:', chain[0], file = sys.stderr)
			chain = chain[1:]
	return ret

def dyadic_link(link, args, flat = False):
	larg, rarg = args
	if depth_match(link, larg, rarg) or flat:
		if hasattr(link, 'conv'):
			return link.conv(link.call, larg, rarg)
		return link.call(larg, rarg)
	if hasattr(link, 'ldepth') and depth(larg) < link.ldepth:
		return dyadic_link(link, ([larg], rarg))
	if hasattr(link, 'rdepth') and depth(rarg) < link.rdepth:
		return dyadic_link(link, (larg, [rarg]))
	if (hasattr(link, 'rdepth') and depth(larg) - depth(rarg) < link.ldepth - link.rdepth) or not hasattr(link, 'ldepth'):
		return [dyadic_link(link, (larg, y)) for y in rarg]
	if (hasattr(link, 'ldepth') and depth(larg) - depth(rarg) > link.ldepth - link.rdepth) or not hasattr(link, 'rdepth'):
		return [dyadic_link(link, (x, rarg)) for x in larg]
	return [x if y == None else y if x == None else dyadic_link(link, (x, y)) for x, y in itertools.zip_longest(*args)]

def eval(string, dirty = True):
	return listify(ast.literal_eval(string), dirty)

def flatten(argument):
	flat = []
	if type(argument) == list:
		for item in argument:
			flat += flatten(item)
	else:
		flat.append(argument)
	return flat

def from_base(digits, base):
	integer = 0
	for digit in digits:
		integer = base * integer + digit
	return integer

def from_exponents(exponents):
	integer = 1
	for index, exponent in enumerate(exponents):
		integer *= sympy.ntheory.generate.prime(index + 1) ** exponent
	return integer

def identity(argument):
	return argument

def iterable(argument, range = False):
	return argument if type(argument) == list else (range(1, argument + 1) if range else [argument])

def index(haystack, needle):
	for index, item in enumerate(haystack):
		if item == needle:
			return 1 + index
	return 0

def isqrt(number):
	a = number
	b = (a + 1) // 2
	while b < a:
		a = b
		b = (a + number // a) // 2
	return int(a)

def last_input():
	if len(sys.argv) > 3:
		return eval(sys.argv[-1])
	return eval(input())

def leading_constant(chain):
	return chain and arities(chain) + [1] < [0, 2] * len(chain)

def listify(iterable, dirty = False):
	if type(iterable) == str and dirty:
		return list(iterable)
	if type(iterable) in (int, float, complex) or (type(iterable) == str and len(iterable) == 1):
		return iterable
	return list(listify(item, dirty) for item in iterable)

def max_arity(links):
	return max(arities(links)) if min(arities(links)) > -1 else ~max(arities(links))

def monadic_chain(chain, arg):
	for link in chain:
		if link.arity < 0:
			link.arity = max(1, link.arity)
	if leading_constant(chain):
		ret = niladic_link(chain[0])
		chain = chain[1:]
	else:
		ret = arg
	while chain:
		if arities(chain[0:2]) == [2, 1]:
			ret = dyadic_link(chain[0], (ret, monadic_link(chain[1], arg)))
			chain = chain[2:]
		elif arities(chain[0:2]) == [2, 0]:
			ret = dyadic_link(chain[0], (ret, niladic_link(chain[1])))
			chain = chain[2:]
		elif arities(chain[0:2]) == [0, 2]:
			ret = dyadic_link(chain[1], (niladic_link(chain[0]), ret))
			chain = chain[2:]
		elif chain[0].arity == 2:
			ret = dyadic_link(chain[0], (ret, arg))
			chain = chain[1:]
		elif chain[0].arity == 1:
			ret = monadic_link(chain[0], ret)
			chain = chain[1:]
		else:
			print('Skipped atom:', chain[0], file = sys.stderr)
			chain = chain[1:]
	return ret

def monadic_link(link, arg, flat = False):
	if depth_match(link, arg) or flat:
		if hasattr(link, 'conv'):
			return link.conv(link.call, arg)
		return link.call(arg)
	if depth(arg) < link.ldepth:
		return monadic_link(link, [arg])
	return [monadic_link(link, z) for z in arg]

def multiset_difference(left, right):
	result = iterable(left)[::-1]
	for element in iterable(right):
		if element in result:
			result.remove(element)
	return result[::-1]

def multiset_intersect(left, right):
	right = iterable(right)[:]
	result = []
	for element in iterable(left):
		if element in right:
			result.append(element)
			right.remove(element)
	return result

def multiset_symdif(left, right):
	return multiset_union(multiset_difference(left, right), multiset_difference(right, left))

def multiset_union(left, right):
	return left + multiset_difference(right, left)

def niladic_chain(chain):
	if not chain or chain[0].arity > 0:
		return monadic_chain(chain, 0)
	return monadic_chain(chain[1:], chain[0].call())

def niladic_link(link):
	return link.call()

def ntimes(links, args, cumulative = False):
	ret, rarg = args
	cumret = []
	for _ in range(variadic_link(links[1], args) if len(links) == 2 else last_input()):
		if cumulative:
			cumret.append(ret)
		larg = ret
		ret = variadic_link(links[0], (larg, rarg))
		rarg = larg
	return cumret + [ret] if cumulative else ret

def overload(operators, *args):
	for operator in operators:
		try:
			ret = operator(*args)
		except:
			pass
		else:
			return ret

def Pi(number):
	if type(number) == int:
		return inf if number < 0 else math.factorial(number)
	return math.gamma(number + 1)

def rld(runs):
	return list(itertools.chain(*[[u] * v for u, v in runs]))

def rotate_left(array, units):
	array = iterable(array)
	length = len(array)
	return array[units % length :] + array[: units % length] if length else []

def sparse(link, args, indices):
	larg = args[0]
	indices = [index - 1 if index > 0 else index - 1 + len(larg) for index in iterable(variadic_link(indices, args))]
	ret = iterable(variadic_link(link, args))
	return [ret[t % len(ret)] if t in indices else u for t, u in enumerate(larg)]

def split_at(iterable, needle):
	chunk = []
	for element in iterable:
		if element == needle:
			yield chunk
			chunk = []
		else:
			chunk.append(element)
	yield chunk

def stringify(iterable, recurse = True):
	if type(iterable) != list:
		return iterable
	if str in map(type, iterable) and not list in map(type, iterable):
		return ''.join(map(str, iterable))
	iterable = [stringify(item) for item in iterable]
	return stringify(iterable, False) if recurse else iterable

def symmetric_mod(number, half_divisor):
	modulus = number % (2 * half_divisor)
	return modulus - 2 * half_divisor * (modulus > half_divisor)

def trim(trimmee, trimmer, left = False, right = False):
	lindex = 0
	rindex = len(trimmee)
	if left:
		while trimmee[lindex] in trimmer and lindex <= rindex:
			lindex += 1
	if right:
		while trimmee[rindex - 1] in trimmer and lindex <= rindex:
			rindex -= 1
	return trimmee[lindex:rindex]

def try_eval(string):
	try:
		return eval(string)
	except:
		return listify(string, True)

def to_base(integer, base):
	digits = []
	integer = abs(integer)
	base = abs(base)
	if base == 0:
		return [integer]
	if base == 1:
		return [1] * integer
	while integer:
		digits.append(integer % base)
		integer //= base
	return digits[::-1] or [0]

def to_exponents(integer):
	if integer == 1:
		return []
	pairs = sympy.ntheory.factor_.factorint(integer)
	exponents = []
	for prime in sympy.ntheory.generate.primerange(2, max(pairs.keys()) + 1):
		if prime in pairs.keys():
			exponents.append(pairs[prime])
		else:
			exponents.append(0)
	return exponents

def unique(iterable):
	result = []
	for element in iterable:
		if not element in result:
			result.append(element)
	return result

def variadic_chain(chain, args):
	args = list(filter(None.__ne__, args))
	if len(args) == 0:
		return niladic_chain(chain)
	if len(args) == 1:
		return monadic_chain(chain, args[0])
	if len(args) == 2:
		return dyadic_chain(chain, args)

def variadic_link(link, args, flat = False):
	if link.arity < 0:
		args = list(filter(None.__ne__, args))
		link.arity = len(args)
	if link.arity == 0:
		return niladic_link(link)
	if link.arity == 1:
		return monadic_link(link, args[0], flat)
	if link.arity == 2:
		return dyadic_link(link, args, flat)

def while_loop(link, condition, args, cumulative = False):
	ret, rarg = args
	cumret = []
	while variadic_link(condition, (ret, rarg)):
		if cumulative:
			cumret.append(ret)
		larg = ret
		ret = variadic_link(link, (larg, rarg))
		rarg = larg
	return cumret + [ret] if cumulative else ret