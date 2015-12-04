import math

def fromBase(digits, base):
	integer = 0
	for digit in digits:
		integer = base * integer + digit
	return integer

def toBase(integer, base):
	digits = []
	integer = abs(integer)
	base = abs(base)
	while integer:
		digits.append(integer % base)
		integer //= base
	return digits[::-1] or [0]