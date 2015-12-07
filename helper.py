import math

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

def symmetric_mod(number, half_divisor):
	modulus = number % (2 * half_divisor)
	return modulus - 2 * half_divisor * (modulus > half_divisor)

def toBase(integer, base):
	digits = []
	integer = abs(integer)
	base = abs(base)
	while integer:
		digits.append(integer % base)
		integer //= base
	return digits[::-1] or [0]