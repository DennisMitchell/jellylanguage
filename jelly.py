import fractions, functools, helper, itertools, operator, sympy

class attrdict(dict):
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)
		self.__dict__ = self

def arities(links):
	return [link.arity for link in links]

def create_chain(chain, arity, depth = -1, ldepth = -1, rdepth = -1):
	return attrdict(
		arity = arity,
		depth = depth,
		ldepth = ldepth,
		rdepth = rdepth,
		call = lambda x = None, y = None: variadic_chain(chain, (x, y))
	)

def create_literal(string):
	return attrdict(
		arity = 0,
		call = lambda: helper.eval(string)
	)

def copy(value):
	atoms['®'].call = lambda: value
	return value

def depth(link):
	if type(link) != list and type(link) != tuple:
		return 0
	return 1 + depth(link[0])

def depth_match(link_depth, arg):
	return link_depth == -1 or link_depth == depth(arg)

def variadic_link(link, args):
	if link.arity < 0:
		args = list(filter(None.__ne__, args))
		link.arity = len(args)
	if link.arity == 0:
		return niladic_link(link)
	if link.arity == 1:
		return monadic_link(link, args[0])
	if link.arity == 2:
		return dyadic_link(link, args)
	raise hell # unimplemented

def variadic_chain(chain, args):
	args = list(filter(None.__ne__, args))
	if len(args) == 0:
		return niladic_chain(chain)
	if len(args) == 1:
		return monadic_chain(chain, args[0])
	if len(args) == 2:
		return dyadic_chain(chain, args)
	raise hell # unimplemented

def niladic_link(link):
	return link.call()

def niladic_chain(chain):
	if len(chain) > 1 and chain[1].arity == 0:
		return dyadic_chain(chain[2:], (chain[0].call(), chain[1].call()))
	return monadic_chain(chain[1:], chain[0].call())

def monadic_link(link, arg):
	if depth_match(link.depth, arg):
		return link.call(arg)
	if depth(arg) < link.depth:
		return monadic_link(link, [arg])
	return [monadic_link(link, z) for z in arg]

def monadic_chain(chain, arg):
	for link in chain:
		if link.arity == -1:
			link.arity = 1
	if chain and arities(chain) < [0, 2] * len(chain):
		ret = niladic_link(chain[0])
		chain = chain[1:]
	else:
		ret = arg
	while chain:
		if arities(chain[0:3]) == [2, 2, 0]:
			ret = dyadic_link(chain[0], (ret, dyadic_link(chain[1], (arg, niladic_link(chain[2])))))
			chain = chain[3:]
		elif arities(chain[0:2]) == [2, 1]:
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
			raise hell # unimplemented
	return ret

def dyadic_link(link, args):
	larg, rarg = args
	if depth_match(link.ldepth, larg) and depth_match(link.rdepth, rarg):
		return link.call(larg, rarg)
	if depth(larg) < link.ldepth:
		return dyadic_link(link, ([larg], rarg))
	if depth(rarg) < link.rdepth:
		return dyadic_link(link, (larg, [rarg]))
	if depth(larg) - depth(rarg) < link.ldepth - link.rdepth:
		return [dyadic_link(link, (larg, y)) for y in rarg]
	if depth(larg) - depth(rarg) > link.ldepth - link.rdepth:
		return [dyadic_link(link, (x, rarg)) for x in larg]
	return [x if y == None else y if x == None else dyadic_link(link, (x, y)) for x, y in itertools.zip_longest(*args)]

def dyadic_chain(chain, args):
	larg, rarg = args
	for link in chain:
		if link.arity == -1:
			link.arity = 2
	if chain and chain[0].arity == 0:
		ret = niladic_link(chain[0])
		chain = chain[1:]
	elif chain and arities(chain) < [0, 2] * len(chain):
		ret = dyadic_link(chain[0], args)
		chain = chain[1:]
	else:
		ret = larg
	while chain:
		if arities(chain[0:3]) == [2, 2, 0]:
			ret = dyadic_link(chain[1], (dyadic_link(chain[0], (ret, rarg)), niladic_link(chain[2])))
			chain = chain[3:]
		elif arities(chain[0:2]) == [2, 2]:
			ret = dyadic_link(chain[0], (ret, dyadic_link(chain[1], args)))
			chain = chain[2:]
		elif arities(chain[0:2]) == [2, 0]:
			ret = dyadic_link(chain[0], (ret, niladic_link(chain[1])))
			chain = chain[2:]
		elif arities(chain[0:2]) == [0, 2]:
			ret = dyadic_link(chain[1], (ret, niladic_link(chain[0])))
			chain = chain[2:]
		elif chain[0].arity == 2:
			ret = dyadic_link(chain[0], (ret, rarg))
			chain = chain[1:]
		elif chain[0].arity == 1:
			ret = monadic_link(chain[0], ret)
			chain = chain[1:]
		else:
			raise hell # unimplemented
	return ret

atoms = {
	'³': attrdict(
		arity = 0,
		call = lambda: 256
	),
	'⁴': attrdict(
		arity = 0,
		call = lambda: 16
	),
	'⁵': attrdict(
		arity = 0,
		call = lambda: 10
	),
	'⁶': attrdict(
		arity = 0,
		call = lambda: ' '
	),
	'⁷': attrdict(
		arity = 0,
		call = lambda: '\n'
	),
	'A': attrdict(
		arity = 1,
		depth = 0,
		call = abs
	),
	'a': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: x and y
	),
	'B': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.to_base(z, 2)
	),
	'Ḅ': attrdict(
		arity = 1,
		depth = 1,
		call = lambda z: helper.from_base(z, 2)
	),
	'b': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: helper.to_base(x, y)
	),
	'ḅ': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 0,
		call = lambda x, y: helper.from_base(x, y)
	),
	'C': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: 1 - z
	),
	'c': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: helper.div(helper.Pi(x), helper.Pi(x - y) * helper.Pi(y))
	),
	'D': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.to_base(z, 10)
	),
	'Ḍ': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.from_base(z, 10)
	),
	'Ḋ': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: z[1:]
	),
	'g': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = fractions.gcd
	),
	'Ɠ': attrdict(
		arity = 0,
		call = lambda: helper.eval(input())
	),
	'H': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.div(z, 2)
	),
	'Ḥ': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: z * 2
	),
	'Ḣ': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: z.pop(0)
	),
	'ḣ': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = 0,
		call = lambda x, y: x[: y - 1]
	),
	'I': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.div(1, z)
	),
	'ị': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = -1,
		call = lambda x, y: y[(x - 1) % len(y)]
	),
	'L': attrdict(
		arity = 1,
		depth = -1,
		call = len
	),
	'l': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: helper.overload((helper.math.log, helper.cmath.log), x, y)
	),
	'N': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: -z
	),
	'O': attrdict(
		arity = 1,
		depth = 1,
		call = lambda z: functools.reduce(operator.add, [[u + 1] * v for u, v in enumerate(z)])
	),
	'o': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: x or y
	),
	'P': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: functools.reduce(lambda x, y: dyadic_link(atoms['×'], (x, y)), z, 1)
	),
	'Ṗ': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: z[:-1]
	),
	'R': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: list(range(1, int(z) + 1) or range(int(z), -int(z) + 1))
	),
	'r': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: list(range(int(x), int(y) + 1) or range(int(x), int(y) - 1, -1))
	),
	'S': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: functools.reduce(lambda x, y: dyadic_link(atoms['+'], (x, y)), z, 0)
	),
	'Ṡ': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: (z > 0) - (z < 0)
	),
	'Ṫ': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: z.pop()
	),
	'ṫ': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = 0,
		call = lambda x, y: x[y - 1 :]
	),
	'U': attrdict(
		arity = 1,
		depth = 1,
		call = lambda z: z[::-1]
	),
	'x': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 1,
		call = lambda x, y: helper.rld(zip(x, y))
	),
	'Z': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: helper.listify(map(lambda z: filter(None.__ne__, z), itertools.zip_longest(*z)))
	),
	'z': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: helper.listify(itertools.zip_longest(*x, fillvalue = y))
	),
	'!': attrdict(
		arity = 1,
		depth = 0,
		call = helper.Pi
	),
	'<': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: int(x < y)
	),
	'=': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: int(x == y)
	),
	'>': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: int(x > y)
	),
	':': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: helper.div(x, y, True)
	),
	',': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: [x, y]
	),
	';': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: (x if depth(x) else [x]) + (y if depth(y) else [y])
	),
	'+': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.add
	),
	'_': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.sub
	),
	'×': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.mul
	),
	'÷': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = helper.div
	),
	'%': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.mod
	),
	'*': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.pow
	),
	'&': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.and_
	),
	'^': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.xor
	),
	'|': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = operator.or_
	),
	'~': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: ~z
	),
	'²': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: z ** 2
	),
	'½': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.sqrt, helper.cmath.sqrt), z)
	),
	'°': attrdict(
		arity = 1,
		depth = 0,
		call = helper.math.radians
	),
	'¬': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: int(not z)
	),
	'‘': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: z + 1
	),
	'’': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: z - 1
	),
	'«': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = min
	),
	'»': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = max
	),
	'©': attrdict(
		arity = 1,
		depth = -1,
		call = copy
	),
	'®': attrdict(
		arity = 0,
		depth = -1
	),
	'{': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: x
	),
	'¹': attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: z
	),
	'}': attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: y
	),
	'ÆA': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.cos, helper.cmath.cos), z)
	),
	'ÆẠ': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.acos, helper.cmath.acos), z)
	),
	'ÆC': attrdict(
		arity = 1,
		depth = 0,
		call = sympy.ntheory.generate.primepi
	),
	'ÆD': attrdict(
		arity = 1,
		depth = 0,
		call = sympy.ntheory.factor_.divisors
	),
	'ÆE': attrdict(
		arity = 1,
		depth = 0,
		call = helper.to_exponents
	),
	'ÆẸ': attrdict(
		arity = 1,
		depth = 1,
		call = helper.from_exponents
	),
	'ÆF': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: [[x, y] for x, y in sympy.ntheory.factor_.factorint(z).items()]
	),
	'Æf': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.rld(sympy.ntheory.factor_.factorint(z).items())
	),
	'ÆN': attrdict(
		arity = 1,
		depth = 0,
		call = sympy.ntheory.generate.prime
	),
	'Æn': attrdict(
		arity = 1,
		depth = 0,
		call = sympy.ntheory.generate.nextprime
	),
	'ÆP': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: int(sympy.primetest.isprime(z))
	),
	'Æp': attrdict(
		arity = 1,
		depth = 0,
		call = sympy.ntheory.generate.prevprime
	),
	'ÆR': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: list(sympy.ntheory.generate.primerange(2, z + 1))
	),
	'ÆT': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.tan, helper.cmath.tan), z)
	),
	'ÆṬ': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.atan, helper.cmath.atan), z)
	),
	'ÆṪ': attrdict(
		arity = 1,
		depth = 0,
		call = sympy.ntheory.factor_.totient
	),
	'ÆS': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.sin, helper.cmath.sin), z)
	),
	'ÆṢ': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: helper.overload((helper.math.asin, helper.cmath.asin), z)
	),
	'Æ²': attrdict(
		arity = 1,
		depth = 0,
		call = lambda z: int(helper.isqrt(z) ** 2 == z)
	),
	'Æ½': attrdict(
		arity = 1,
		depth = 0,
		call = helper.isqrt
	),
	'Æ°': attrdict(
		arity = 1,
		depth = 0,
		call = helper.math.degrees
	),
	'æA': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = helper.math.atan2
	),
	'æ%': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = helper.symmetric_mod
	),
}

actors = {
	'ß': lambda index, links: attrdict(
		arity = -1,
		depth = -1,
		ldepth = -1,
		rdepth = -1,
		call = lambda x = None, y = None: variadic_chain(links[index], (x, y))
	),
	'¢': lambda index, links: attrdict(
		arity = 0,
		call = lambda: niladic_chain(links[index - 1])
	),
	'Ç': lambda index, links: attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: monadic_chain(links[index - 1], z)
	),
	'ç': lambda index, links: attrdict(
		arity = 1,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: monadic_chain(links[index - 1], (x, y))
	),
	'Ñ': lambda index, links: attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: monadic_chain(links[(index + 1) % len(links)], z)
	),
	'ñ': lambda index, links: attrdict(
		arity = 1,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: dyadic_chain(links[(index + 1) % len(links)], (x, y))
	)
}

hypers = {
	'@': lambda link, none = None: attrdict(
		arity = 2,
		ldepth = link.rdepth,
		rdepth = link.ldepth,
		call = lambda x, y: link.call(y, x)
	),
	'/': lambda link, none = None: attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: functools.reduce(lambda x, y: dyadic_link(link, (x, y)), z)
	),
	'\\': lambda link, none = None: attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: list(itertools.accumulate(z, lambda x, y: dyadic_link(link, (x, y))))
	),
	'£': lambda index, links: attrdict(
		arity = index.arity,
		depth = -1,
		ldepth = -1,
		rdepth = -1,
		call = lambda x = None, y = None: niladic_chain(links[(variadic_link(index, (x, y)) - 1) % (len(links) - 1)])
	),
	'Ŀ': lambda index, links: attrdict(
		arity = max(1, index.arity),
		depth = -1,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y = None: monadic_chain(links[(variadic_link(index, (x, y)) - 1) % (len(links) - 1)], x)
	),
	'ŀ': lambda index, links: attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: dyadic_chain(links[(dyadic_link(index, (x, y)) - 1) % (len(links) - 1)], (x, y))
	)
}

joints = {
	'¤': lambda links: attrdict(
		arity = 0,
		call = lambda: niladic_chain(links)
	),
	'$': lambda links: attrdict(
		arity = 1,
		depth = -1,
		call = lambda z: monadic_chain(links, z)
	),
	'¥': lambda links: attrdict(
		arity = 2,
		ldepth = -1,
		rdepth = -1,
		call = lambda x, y: dyadic_chain(links, (x, y))
	),
	'¡': lambda links: attrdict(
		arity = max(links[0].arity, links[1].arity),
		depth = -1,
		ldepth = -1,
		rdepth = -1,
		call = lambda x = None, y = None: helper.ntimes(links[0], links[1], (x, y))
	)
}