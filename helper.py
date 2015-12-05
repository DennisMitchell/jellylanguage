import math, sympy

inf = float('inf')

def fromBase(digits, base):
	integer = 0
	for digit in digits:
		integer = base * integer + digit
	return integer

def div(dividend, divisor):
	if divisor == 0:
		return inf
	if divisor == inf:
		return 0
	if type(dividend) == int and type(divisor) == int and not dividend % divisor:
		return dividend // divisor
	return dividend / divisor

def pi(integer):
	if type(integer) == int:
		return inf if integer < 0 else math.factorial(integer)
	return math.gamma(integer + 1)

def prime(integer, function):
	if function == 0:
		return int(sympy.primetest.isprime(integer))
	if function == 1:
		return sympy.ntheory.generate.prime(integer)
	if function == 2:
		return sympy.ntheory.generate.nextprime(integer)
	if function == 3:
		return sympy.ntheory.generate.prevprime(integer)
	if function == 4:
		return sympy.ntheory.generate.primepi(integer)
	if function == 5:
		return list(sympy.ntheory.generate.primerange(0, integer + 1))

def toBase(integer, base):
	digits = []
	integer = abs(integer)
	base = abs(base)
	while integer:
		digits.append(integer % base)
		integer //= base
	return digits[::-1] or [0]