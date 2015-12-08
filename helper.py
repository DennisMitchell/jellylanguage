import functools, math, operator, sympy

inf = float('inf')

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
		return inf
	if divisor == inf:
		return 0
	if floor or (type(dividend) == int and type(divisor) == int and not dividend % divisor):
		return dividend // divisor
	return dividend / divisor

def isqrt(number):
	a = number
	b = (a + 1) // 2
	while b < a:
		a = b
		b = (a + number // a) // 2
	return a

def pi(number):
	if type(number) == int:
		return inf if number < 0 else math.factorial(number)
	return math.gamma(number + 1)

def rld(runs):
	return functools.reduce(operator.add, [[u] * v for u, v in runs])

def symmetric_mod(number, half_divisor):
	modulus = number % (2 * half_divisor)
	return modulus - 2 * half_divisor * (modulus > half_divisor)

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