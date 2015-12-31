import ast, cmath, functools, itertools, jelly, math, sympy

inf = float('inf')
nan = float('nan')

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

def div(dividend, divisor, floor = False):
	if divisor == 0:
		return nan if dividend == 0 else inf
	if divisor == inf:
		return 0
	if floor or (type(dividend) == int and type(divisor) == int and not dividend % divisor):
		return int(dividend // divisor)
	return dividend / divisor

def eval(string):
	return listify(ast.literal_eval(string))

def identity(argument):
	return argument

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

def listify(iterable):
	if not type(iterable) in (int, float, complex, str):
		iterable = list(map(listify, iterable))
	return iterable

def ntimes(link, repetitions, args):
	ret, rarg = args
	for _ in range(jelly.variadic_link(repetitions, args)):
		larg = ret
		ret = jelly.variadic_link(link, (larg, rarg))
		rarg = larg
	return ret

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

def symmetric_mod(number, half_divisor):
	modulus = number % (2 * half_divisor)
	return modulus - 2 * half_divisor * (modulus > half_divisor)

def try_eval(string):
	try:
		return eval(string)
	except:
		return string

def to_base(integer, base):
	digits = []
	integer = abs(integer)
	base = abs(base)
	while integer:
		digits.append(integer % base)
		integer //= base
	return digits[::-1] or [0]

def to_exponents(integer):
	pairs = sympy.ntheory.factor_.factorint(integer)
	exponents = []
	for prime in sympy.ntheory.generate.primerange(2, max(pairs.keys()) + 1):
		if prime in pairs.keys():
			exponents.append(pairs[prime])
		else:
			exponents.append(0)
	return exponents