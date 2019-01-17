import cmath, copy, functools, hashlib, itertools, locale, math, operator, re, sys, time

from .utils import attrdict, lazy_import

random, sympy, urllib_request = lazy_import('random sympy urllib.request')

code_page  = '''¡¢£¤¥¦©¬®µ½¿€ÆÇÐÑ×ØŒÞßæçðıȷñ÷øœþ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~¶'''
code_page += '''°¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ƁƇƊƑƓƘⱮƝƤƬƲȤɓƈɗƒɠɦƙɱɲƥʠɼʂƭʋȥẠḄḌẸḤỊḲḶṂṆỌṚṢṬỤṾẈỴẒȦḂĊḊĖḞĠḢİĿṀṄȮṖṘṠṪẆẊẎŻạḅḍẹḥịḳḷṃṇọṛṣṭ§Äẉỵẓȧḃċḋėḟġḣŀṁṅȯṗṙṡṫẇẋẏż«»‘’“”'''

# Unused symbols for single-byte atoms/quicks: (quƁƘȤɦɱɲƥʠʂȥḥḳṇẉỵẓėġṅẏ

str_digit = '0123456789'
str_lower = 'abcdefghijklmnopqrstuvwxyz'
str_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

inf = float('inf')
nan = float('nan')
sys.setrecursionlimit(1 << 30)

def arities(links):
	return [link.arity for link in links]

def at_index(index, array):
	array = iterable(array)
	if not array:
		return 0
	low_index = math.floor(index) - 1
	high_index = math.ceil(index) - 1
	if low_index == high_index:
		return array[low_index % len(array)]
	return [array[low_index % len(array)], array[high_index % len(array)]]

def at_index_ndim(indices, array):
	retval = array
	for index in indices:
		retval = at_index(index, retval)
	return retval

def base_decompression(integer, digits):
	digits = iterable(digits, make_range=True)
	return [digits[i-1] for i in to_base(integer, len(digits))]

def bounce(array):
	return array[:-1] + array[::-1]

def carmichael(n):
	n = int(n)
	if n < 1:
		return 0
	c = 1
	for p, k in sympy.ntheory.factor_.factorint(n).items():
		c = lcm(c, 2 ** (k - 2) if p == 2 < k else (p - 1) * p ** (k - 1))
	return c

def create_chain(chain, arity = -1, isForward = True):
	return attrdict(
		arity = arity,
		chain = chain,
		call = lambda x = None, y = None: variadic_chain(chain, isForward and (x, y) or (y, x))
	)

def create_literal(string):
	return attrdict(
		arity = 0,
		call = lambda: python_eval(string, False)
	)

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

def convolve(left, right):
	left, right = iterable(left, make_range = True), iterable(right, make_range = True)
	result = [0]*(len(left)+len(right)-1)
	for i,x in enumerate(left):
		for j,y in enumerate(right):
			result[i+j] += x*y
	return result

def convolve_power(polynomial, exponent):
	retval = [1]
	for _ in range(exponent):
		retval = convolve(retval, polynomial)
	return retval

def copy_to(atom, value):
	atom.call = lambda: value
	return value

def determinant(matrix):
	matrix = sympy.Matrix(matrix)
	if matrix.is_square:
		return simplest_number(matrix.det())
	return simplest_number(math.sqrt((matrix * matrix.transpose()).det()))

def div(dividend, divisor, floor = False):
	if divisor == 0:
		return dividend * inf
	if divisor == inf:
		return 0
	if floor or (type(dividend) == int and type(divisor) == int and not dividend % divisor):
		return int(dividend // divisor)
	return dividend / divisor

def depth(link):
	if type(link) != list:
		return 0
	if not link:
		return 1
	return 1 + max(map(depth, link))

def diagonals(matrix):
	shifted = [None] * len(matrix)
	for index, row in enumerate(map(iterable, reversed(matrix))):
		shifted[~index] = index * [None] + row
	return rotate_left(zip_ragged(shifted), len(matrix) - 1)

def distinct_sieve(array):
	array = iterable(array, make_digits = True)
	result = []
	for (i, x) in enumerate(array):
		result.append(1 if i == array.index(x) else 0)
	return result

def dot_product(left, right, truncate = False):
	left, right = iterable(left), iterable(right)
	if complex in map(type, left + right):
		right = [complex(t).conjugate() for t in right]
	if truncate:
		product = sum(map(operator.mul, left, right))
	else:
		product = sum(dyadic_link(atoms['×'], (left, right)))
	if product.imag == 0:
		product = product.real
		if type(product) != int and product.is_integer():
			product = int(product)
	return product

def dyadic_chain(chain, args):
	larg, rarg = args
	for link in chain:
		if link.arity < 0:
			link.arity = 2
	if chain and arities(chain[0:3]) == [2, 2, 2]:
		ret = dyadic_link(chain[0], args)
		chain = chain[1:]
	elif leading_nilad(chain):
		ret = niladic_link(chain[0])
		chain = chain[1:]
	else:
		ret = larg
	while chain:
		if arities(chain[0:3]) == [2, 2, 0] and leading_nilad(chain[2:]):
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
			output(ret)
			ret = niladic_link(chain[0])
			chain = chain[1:]
	return ret

def dyadic_link(link, args, conv = True, lflat = False, rflat = False):
	larg, rarg = args
	lflat = lflat or not hasattr(link, 'ldepth')
	rflat = rflat or not hasattr(link, 'rdepth')
	larg_depth = lflat or depth(larg)
	rarg_depth = rflat or depth(rarg)
	if (lflat or link.ldepth == larg_depth) and (rflat or link.rdepth == rarg_depth):
		if conv and hasattr(link, 'conv'):
			return link.conv(link.call, larg, rarg)
		return link.call(larg, rarg)
	conv = conv and hasattr(link, 'conv')
	if not lflat and larg_depth < link.ldepth:
		return dyadic_link(link, ([larg], rarg))
	if not rflat and rarg_depth < link.rdepth:
		return dyadic_link(link, (larg, [rarg]))
	if not rflat and (lflat or larg_depth - rarg_depth < link.ldepth - link.rdepth):
		return [dyadic_link(link, (larg, y)) for y in rarg]
	if not lflat and (rflat or larg_depth - rarg_depth > link.ldepth - link.rdepth):
		return [dyadic_link(link, (x, rarg)) for x in larg]
	return [dyadic_link(link, (x, y)) for x, y in zip(*args)] + larg[len(rarg) :] + rarg[len(larg) :]

def enumerate_md(array, upper_level = []):
	for i, item in enumerate(array):
		if type(item) != list:
			yield [upper_level + [i + 1], item]
		else:
			yield from enumerate_md(item, upper_level + [i + 1])

def equal(array):
	array = iterable(array)
	return int(all(item == array[0] for item in array))

def extremes(min_or_max, link, array):
	x,y = array
	x = iterable(x, make_range=True)
	if not x:
		return []
	results = [variadic_link(link, (t, y)) for t in x]
	best = min_or_max(results)
	return [t for t, ft in zip(x, results) if ft == best]

def filter_array(sand, mesh, is_in = True):
	mesh = {repr(element) for element in iterable(mesh)}
	return [element for element in iterable(sand) if (repr(element) in mesh) == is_in]

def flatten(argument):
	flat = []
	if type(argument) == list:
		for item in argument:
			flat += flatten(item)
	else:
		flat.append(argument)
	return flat

def foldl(*args):
	return reduce(*args, arity = 2)

def from_base(digits, base):
	integer = 0
	for digit in digits:
		integer = base * integer + digit
	return integer

def from_diagonals(diagonals):
	upper_right = 1
	while len(diagonals[upper_right - 1]) > 1:
		upper_right += 1
	diagonals = rotate_left(diagonals, upper_right)
	shift = len(diagonals) - upper_right
	index = 0
	while shift:
		diagonals[index] = shift * [None] + diagonals[index]
		index += 1
		shift -= 1
	return zip_ragged(diagonals)

def from_exponents(exponents):
	integer = 1
	for index, exponent in enumerate(exponents):
		integer *= sympy.ntheory.generate.prime(index + 1) ** exponent
	return integer

def from_factorial_base(digits):
	placeValue = 1
	integer = 0
	for nextPlaceIndex, digit in enumerate(digits[::-1], 1):
		integer += digit * placeValue
		placeValue *= nextPlaceIndex
	return integer

def from_primorial_base(digits):
	integer = digits and digits[-1] or 0
	for placeIndex, digit in enumerate(digits[-2::-1], 1):
		integer += digit * sympy.ntheory.generate.primorial(placeIndex)
	return integer

def simplest_number(number):
	if abs(number ** 2) != number ** 2:
		return number
	if number % 1:
		return float(number)
	return int(number)

def get_request(url):
	url = ''.join(map(str, url))
	url = (re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", url) == None and "http://" or "") + url
	response = urllib_request.request.urlopen(url).read()
	try:
		return list(response.decode('utf-8'))
	except:
		return list(response.decode('latin-1'))

def grid(array):
	if depth(array) == 1:
		return join(array, ' ')
	if depth(array) == 2 and equal(map(len, array)):
		array = [[str(entry) for entry in row] for row in array]
		width = max(max([len(entry) for entry in row]) if row else 0 for row in array)
		array = [[list(entry.rjust(width)) for entry in row] for row in array]
		return join([join(row, ' ') for row in array], '\n')
	if depth(array) == 3 and all(type(item) == str for item in flatten(array)):
		array = [[''.join(entry) for entry in row] for row in array]
		width = max(max([len(entry) for entry in row]) if row else 0 for row in array)
		array = [[list(entry.ljust(width)) for entry in row] for row in array]
		return join([join(row, ' ') for row in array], '\n')
	return join(array, '\n')

def group(array):
	array = iterable(array, make_digits = True)
	grouped = {}
	for index, item in enumerate(array):
		item = repr(item)
		if item in grouped:
			grouped[item].append(index + 1)
		else:
			grouped[item] = [index + 1]
	try:
		return [grouped[key] for key in sorted(grouped, key = eval)]
	except TypeError:
		return [grouped[key] for key in sorted(grouped)]

def group_md(array):
	array = iterable(array, make_digits = True)
	grouped = {}
	for index, item in enumerate_md(array):
		item = repr(item)
		if item in grouped:
			grouped[item].append(index)
		else:
			grouped[item] = [index]
	try:
		return [grouped[key] for key in sorted(grouped, key = eval)]
	except TypeError:
		return [grouped[key] for key in sorted(grouped)]

def group_equal(array):
	array = iterable(array, make_digits = True)
	groups = []
	for x in array:
		if groups and groups[-1][0] == x:
			groups[-1].append(x)
		else:
			groups.append([x])
	return groups

def group_lengths(array):
	array = iterable(array, make_digits = True)
	lengths = []
	previous_item = None
	for x in array:
		if lengths and previous_item == x:
			lengths[-1] += 1
		else:
			lengths.append(1)
		previous_item = x
	return lengths

def identity(argument):
	return argument

def iterable(argument, make_copy = False, make_digits = False, make_range = False):
	the_type = type(argument)
	if the_type == list:
		return copy.deepcopy(argument) if make_copy else argument
	if the_type != str and make_digits:
		return to_base(argument, 10)
	if the_type != str and make_range:
		return list(range(1, int(argument) + 1))
	return [argument]

def index_of(haystack, needle):
	for index, item in enumerate(iterable(haystack)):
		if item == needle:
			return 1 + index
	return 0

def index_of_md(haystack, needle):
	for index, item in enumerate_md(haystack):
		if item == needle:
			return index
	return []

def indices_md(array, upper_level = []):
	a_indices = []
	for i, item in enumerate(array):
		if type(item) != list:
			a_indices.append(upper_level + [i + 1])
		else:
			a_indices.extend(indices_md(item, upper_level = upper_level + [i + 1]))
	return a_indices

def isqrt(number):
	a = number
	b = (a + 1) // 2
	while b < a:
		a = b
		b = (a + number // a) // 2
	return int(a)

def is_palindrome(argument):
	argument = iterable(argument, make_digits = True)
	return int(argument == argument[::-1])

def is_string(argument):
	if type(argument) != list:
		return False
	return all(map(lambda t: type(t) == str, argument))

def jelly_eval(code, arguments):
	return variadic_chain(parse_code(code)[-1] if code else '', arguments)

def jelly_hash(spec, object):
	if len(spec) > 2:
		spec = [spec[0], spec[1:]]
	if type(spec[0]) == list:
		digits = [code_page.find(item) if type(item) == str else item for item in spec[0]]
		magic = from_base([digit + 1 for digit in digits], 250) + 1
	else:
		magic = spec[0] + 1

	buckets = spec[1] if type(spec[1]) == list else range(1, spec[1] + 1)

	shake = hashlib.shake_256(repr(object).encode('utf-8')).digest(512)
	longs = [int.from_bytes(shake[i : i + 8], 'little') for i in range(0, 512, 8)]
	temp = sum(((magic >> i) - (magic >> i + 1)) * longs[i] for i in range(64))
	hash = temp % 2**64 * len(buckets) >> 64
	return buckets[hash - 1]

def jelly_uneval(argument, top = True):
	the_type = type(argument)
	if the_type in (float, int):
		return jelly_uneval_real(argument)
	if the_type == complex:
		return jelly_uneval_real(argument.real) + 'ı' + jelly_uneval_real(argument.imag)
	if the_type == str:
		return '”' + argument
	if all(map(is_string, argument)):
		strings = [''.join(string) for string in argument]
		if all(map(lambda t: code_page.find(t) < 250, ''.join(strings))):
			return '“' + '“'.join(strings) + '”'
	if is_string(argument):
		string = ''.join(argument)
		if all(map(lambda t: code_page.find(t) < 250, string)):
			return '“' + string + '”'
	middle = ','.join(jelly_uneval(item, top = False) for item in argument)
	return middle if top else '[' + middle + ']'

def jelly_uneval_real(number):
	string = str(number if number % 1 else int(number))
	return string.lstrip('0') if number else string

def jellify(element, dirty = False):
	if element is None:
		return []
	if type(element) == str and dirty:
		return list(element)
	if type(element) in (int, float, complex) or (type(element) == str and len(element) == 1):
		return element
	try:
		return [jellify(item, dirty) for item in element]
	except:
		if element.is_integer:
			return int(element)
		if element.is_real:
			return float(element)
		return complex(element)

def join(array, glue):
	array = iterable(array, make_copy = True)
	last = array.pop() if array else []
	glue = iterable(glue)
	ret = []
	for item in array:
		ret += iterable(item) + glue
	return ret + iterable(last)

def last_input():
	if len(sys.argv) > 3:
		return python_eval(sys.argv[-1])
	return python_eval(input())

def leading_nilad(chain):
	return chain and arities(chain) + [1] < [0, 2] * len(chain)

def lcm(x, y):
	return x * y // (math.gcd(x, y) or 1)

def loop_until_loop(link, args, return_all = False, return_loop = False, vary_rarg = True):
	ret, rarg = args
	cumret = []
	while True:
		cumret.append(ret)
		larg = ret
		ret = variadic_link(link, (larg, rarg))
		if vary_rarg:
			rarg = larg
		if ret in cumret:
			if return_all:
				return cumret
			if return_loop:
				return cumret[index_of(cumret, ret) - 1 :]
			return larg

def nfind(links, args):
	larg, rarg = args
	larg = larg or 0
	matches = variadic_link(links[1], args) if len(links) == 2 else last_input()
	found = []
	while len(found) < matches:
		if variadic_link(links[0], (larg, rarg)):
			found.append(larg)
		larg += 1
	return found

def matrix_to_list(matrix):
	return [[simplest_number(entry) for entry in row] for row in matrix.tolist()]

def max_arity(links):
	return max(arities(links)) if min(arities(links)) > -1 else (~max(arities(links)) or -1)

def maximal_indices(iterable):
	maximum = max(iterable or [0])
	return [u + 1 for u, v in enumerate(iterable) if v == maximum]

def maximal_indices_md(iterable, upper_level = [], maximum = None):
	if maximum == None:
		maximum = max(flatten(iterable) or [0])
	result = []
	for i, item in enumerate(iterable):
		if type(item) != list:
			if item == maximum:
				result.append(upper_level + [i + 1])
		else:
			result.extend(maximal_indices_md(item, upper_level = upper_level + [i + 1], maximum = maximum))
	return result

def median(array):
	array = sorted(array)
	return div(array[(len(array) - 1) // 2] + array[len(array) // 2], 2)

def mode(array):
	frequencies = {}
	maxfreq = 0
	retval = []
	for element in array:
		string = repr(element)
		frequencies[string] = frequencies.get(string, 0) + 1
		maxfreq = max(frequencies[string], maxfreq)
	for element in array:
		string = repr(element)
		if frequencies[string] == maxfreq:
			retval.append(element)
			frequencies[string] = 0
	return retval

def modinv(a, m):
	i, _, g = sympy.numbers.igcdex(a, m)
	return i % m if g == 1 else 0

def modulus(dividend, divisor):
	try:
		return dividend % divisor
	except:
		return nan

def mold(content, shape):
	for index in range(len(shape)):
		if type(shape[index]) == list:
			mold(content, shape[index])
		else:
			item = content.pop(0)
			shape[index] = item
			content.append(item)
	return shape

def monadic_chain(chain, arg):
	init = True
	ret = arg
	larg_save = atoms['⁸'].call
	while True:
		if init:
			for link in chain:
				if link.arity < 0:
					link.arity = 1
			if leading_nilad(chain):
				ret = niladic_link(chain[0])
				chain = chain[1:]
			init = False
		if not chain:
			break
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
			if not chain[1:] and hasattr(chain[0], 'chain'):
				arg = ret
				chain = chain[0].chain
				atoms['⁸'].call = lambda literal = arg: literal
				init = True
			else:
				ret = monadic_link(chain[0], ret)
				chain = chain[1:]
		else:
			output(ret)
			ret = niladic_link(chain[0])
			chain = chain[1:]
	atoms['⁸'].call = larg_save
	return ret

def monadic_link(link, arg, flat = False, conv = True):
	flat = flat or not hasattr(link, 'ldepth')
	arg_depth = flat or depth(arg)
	if flat or link.ldepth == arg_depth:
		if conv and hasattr(link, 'conv'):
			return link.conv(link.call, arg)
		return link.call(arg)
	conv = conv and hasattr(link, 'conv')
	if link.ldepth > arg_depth:
		return monadic_link(link, [arg], conv = conv)
	return [monadic_link(link, z, conv = conv) for z in arg]

def multiset_difference(left, right):
	result = iterable(left)[::-1]
	for element in iterable(right):
		if element in result:
			result.remove(element)
	return result[::-1]

def multiset_intersect(left, right):
	right = iterable(right, make_copy = True)
	result = []
	for element in iterable(left):
		if element in right:
			result.append(element)
			right.remove(element)
	return result

def multiset_symdif(left, right):
	return multiset_union(multiset_difference(left, right), multiset_difference(right, left))

def multiset_union(left, right):
	return iterable(left) + multiset_difference(right, left)

def nCr(left, right):
	if type(left) == int and type(right) == int:
		if right < 0:
			right = left - right
		if right < 0 or (left > 0 and right > left):
			return 0
		if left > 0:
			right = min(right, left - right)
		result = 1
		for i in range(right):
			result = result * (left - i) // (i + 1)
		return result
	return div(Pi(left), Pi(left - right) * Pi(right))

def niladic_chain(chain):
	while len(chain) == 1 and hasattr(chain[0], 'chain'):
		chain = chain[0].chain
	if not chain or chain[0].arity > 0:
		return monadic_chain(chain, 0)
	return monadic_chain(chain[1:], chain[0].call())

def niladic_link(link):
	return link.call()

def ntimes(links, args, cumulative = False):
	ret, rarg = args
	repetitions = variadic_link(links[1], args) if len(links) == 2 else last_input()
	repetitions = overload((int, bool), repetitions)
	if cumulative:
		cumret = [0] * repetitions
	for index in range(repetitions):
		if cumulative:
			cumret[index] = ret
		larg = ret
		ret = variadic_link(links[0], (larg, rarg))
		rarg = larg
	return cumret + [ret] if cumulative else ret

def odd_even(array):
	array = iterable(array, make_range = True)
	return [[t for t in array[::2]], [t for t in array[1::2]]]

def order(number, divisor):
	if number == 0 or abs(divisor) == 1:
		return inf
	if divisor == 0:
		return 0
	ret = 0
	while True:
		number, residue = divmod(number, divisor)
		if residue:
			break
		ret += 1
	return ret

def overload(operators, *args):
	for operator in operators:
		try:
			ret = operator(*args)
		except:
			pass
		else:
			return ret

def integer_partitions(n, I=1):
	result = [[n,]]
	for i in range(I, n//2 + 1):
		for p in integer_partitions(n-i, i):
			result.append([i,] + p)
	return result

def neighbors(links, array):
	array = iterable(array, make_digits = True)
	chain = dyadic_chain if links[-1].arity == 2 else monadic_chain
	return [chain(links, list(pair)) for pair in zip(array, array[1:])]

def partitions(array):
	array = iterable(array, make_digits = True)
	ret = []
	for index in range(1, len(array)):
		for subarray in partitions(array[index:]):
			subarray.insert(0, array[:index])
			ret.append(subarray)
	ret.append([array])
	return ret

def parse_code(code):
	lines = regex_flink.findall(code)
	links = [[] for line in lines]
	for index, line in enumerate(lines):
		chains = links[index]
		for word in regex_chain.findall(line):
			chain = []
			arity, start, isForward = chain_separators.get(word[:1], default_chain_separation)
			for token in regex_token.findall(start + word):
				if token in atoms:
					chain.append(atoms[token])
				elif token in quicks:
					popped = []
					while not quicks[token].condition(popped) and (chain or chains):
						popped.insert(0, chain.pop() if chain else chains.pop())
					chain += quicks[token].quicklink(popped, links, index)
				elif token in hypers:
					x = chain.pop() if chain else chains.pop()
					chain.append(hypers[token](x, links))
				else:
					chain.append(create_literal(regex_liter.sub(parse_literal, token)))
			chains.append(create_chain(chain, arity, isForward))
	return links

def parse_literal(literal_match):
	literal = literal_match.group(0)
	if literal[0] in '”⁾':
		return repr(literal[1:].replace('¶', '\n'))
	elif literal[0] == '“':
		if literal[-1] in '«»‘’”':
			mode = literal[-1]
			literal = literal[:-1]
		else:
			mode = ''
		parsed = literal.split('“')[1:]
		if   mode == '»':
			parsed = [sss(string).replace('¶', '\n') for string in parsed]
		elif mode == '‘':
			parsed = [[code_page.find(char) for char in string] for string in parsed]
		elif mode == '’':
			parsed = [from_base([code_page.find(char) + 1 for char in string], 250) for string in parsed]
		else:
			parsed = [string.replace('¶', '\n') for string in parsed]
		if mode not in '‘’':
			parsed = [[string] if len(string) == 1 else string for string in parsed]
		if len(parsed) == 1:
			parsed = parsed[0]
	elif literal[0] == '⁽':
		parsed = from_base([code_page.find(char) + 1 for char in literal[1:]], 250)
		parsed += parsed > 31500 and -62850 or 750
	else:
		parsed = eval('+ 1j *'.join([
			repr(eval('* 10 **'.join(['-1' if part == '-' else (part + '5' if part[-1:] == '.' else part) or repr(2 * index + 1)
			for index, part in enumerate(component.split('ȷ'))])) if component else index)
			for index, component in enumerate(literal.split('ı'))
		]))
	return repr(parsed) + ' '

def partition_at(booleans, array, border = 1):
	booleans = iterable(booleans)
	array = iterable(array)
	chunks = [[], []]
	index = 0
	while index < len(array):
		if index < len(booleans) and booleans[index]:
			chunks.append([])
			chunks[-border].append(array[index])
		else:
			chunks[-1].append(array[index])
		index += 1
	return chunks[1:]

def pemutation_at_index(index, array = None):
	result = []
	if array is None:
		divisor = 1
		count = 0
		while divisor < index:
			count += 1
			divisor *= count
		values = list(range(1, count + 1))
	else:
		values = iterable(array, make_copy = True, make_range = True)
		try:
			divisor = math.factorial(len(values))
		except:
			divisor = functools.reduce(operator.mul, range(1, len(values) + 1), 1)
	index -= 1
	index %= divisor
	while values:
		divisor //= len(values)
		quotient, index = divmod(index, divisor)
		result.append(values.pop(quotient))
	return result

def permutation_index(array):
	result = 1
	array = iterable(array)
	length = len(array)
	for index in range(length):
		k = sum(1 for value in array[index + 1:] if value < array[index])
		try:
			factor = math.factorial(length - index - 1)
		except:
			factor = functools.reduce(operator.mul, range(1, length - index), 1)
		result += k * factor
	return result

def Pi(number):
	if type(number) == int:
		if number < 0:
			return inf
		try:
			return math.factorial(number)
		except:
			return functools.reduce(operator.mul, range(1, number + 1), 1)
	return math.gamma(number + 1)

def powerset(array):
	array = iterable(array, make_range = True)
	ret = []
	for t in range(len(array) + 1):
		ret += jellify(itertools.combinations(array, t))
	return ret

def prefix(links, outmost_links, index):
	ret = [attrdict(arity = max(1, links[0].arity))]
	if len(links) == 1:
		ret[0].call = lambda x, y = None: [variadic_link(links[0], (t, y)) for t in split_prefix(x)]
	else:
		ret[0].call = lambda x, y = None: [variadic_link(links[0], (t, y)) for t in split_rolling(x, niladic_link(links[1]))]
	return ret

def primerange(start, end):
	if start > end:
		return list(sympy.primerange(end, start + 1))[::-1]
	else:
		return list(sympy.primerange(start, end + 1))

def python_eval(string, dirty = True):
	try:
		return jellify(eval(string), dirty)
	except SyntaxError:
		exec(string)
		return []

def quickchain(arity, min_length):
	return attrdict(
		condition =
			(lambda links: len(links) >= min_length and links[0].arity == 0)
			if arity == 0 else
			lambda links:
				len(links) - sum(map(leading_nilad, split_suffix(links)[:-1])) >= min_length,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = arity,
			call = lambda x = None, y = None: variadic_chain(links, (x, y))
		)]
	)

def random_int(pool):
	if not pool:
		return 0
	if type(pool) == list:
		return random.choice(pool)
	return random.randint(1, pool)

def reduce(links, outmost_links, index, arity = 1):
	ret = [attrdict(arity = arity)]
	if len(links) == 1:
		ret[0].call = lambda x, *y: reduce_simple(x, links[0], *y)
	else:
		ret[0].call = lambda x, *y: [reduce_simple(t, links[0], *y) for t in split_fixed(iterable(x), links[1].call())]
	return ret

def reduce_simple(array, link, *init):
	array = iterable(array)
	return functools.reduce(lambda x, y: dyadic_link(link, (x, y)), array, *init)

def reduce_cumulative(links, outmost_links, index):
	ret = [attrdict(arity = 1)]
	if len(links) == 1:
		ret[0].call = lambda t: list(itertools.accumulate(iterable(t), lambda x, y: dyadic_link(links[0], (x, y))))
	else:
		ret[0].call = lambda z: [reduce_simple(t, links[0]) for t in split_rolling(iterable(z), links[1].call())]
	return ret

def rld(runs):
	return list(itertools.chain(*[[u] * v for u, v in runs]))

def rle(array):
	return [[group[0], len(group)] for group in group_equal(array)]

def rotate_left(array, units):
	array = iterable(array)
	length = len(array)
	return array[units % length :] + array[: units % length] if length else []

def shift_left(number, bits):
	if type(number) == int and type(bits) == int:
		return number << bits
	return number * 2 ** bits

def shift_right(number, bits):
	if type(number) == int and type(bits) == int:
		return number >> bits
	return div(number, 2 ** bits, floor = True)

def shuffle(array):
	array = iterable(array, make_copy = True, make_range = True)
	random.shuffle(array)
	return array

def sparse(link, args, indices, indices_literal = False):
	larg = args[0]
	if not indices_literal:
		indices = iterable(variadic_link(indices, args))
	indices = [index - 1 if index > 0 else index - 1 + len(larg) for index in indices]
	ret = iterable(variadic_link(link, args))
	return [ret[t % len(ret)] if t in indices else u for t, u in enumerate(larg)]

def split_around(array, needle):
	chunk = []
	window = len(needle)
	index = 0
	while index < len(array):
		if array[index : index + window] == needle:
			yield chunk
			chunk = []
			index += window
		else:
			chunk.append(array[index])
			index += 1
	yield chunk

def split_at(array, needle):
	chunk = []
	for element in array:
		if element == needle:
			yield chunk
			chunk = []
		else:
			chunk.append(element)
	yield chunk

def split_evenly(array, chunks):
	array = iterable(array)
	min_width, overflow = divmod(len(array), chunks)
	ret = []
	high = 0
	for index in range(chunks):
		low = high
		high = low + min_width + (index < overflow)
		ret.append(array[low : high])
	return ret

def split_fixed(array, width):
	if width < 0:
		return split_fixed_out(array, -width)
	array = iterable(array)
	return [array[index : index + width] for index in range(0, len(array), width)]

def split_fixed_out(array, width):
	array = iterable(array)
	return [array[:index] + array[index + width:] for index in range(0, len(array), width)]

def split_key(control, data):
	groups = {}
	order = []
	count = 0
	for key, item in zip(control, data):
		key = repr(key) if type(key) == list else key
		if key not in groups:
			order.append(key)
			groups[key] = []
		groups[key].append(item)
		count += 1
	result = [groups[key] for key in order]
	if count < len(data):
		result.append(data[count:])
	return result

def split_once(array, needle):
	array = iterable(array, make_digits = True)
	index = index_of(array, needle)
	return [array[0 : index - 1], array[index :]] if index else [array]

def split_prefix(array):
	array = iterable(array)
	return [array[:index + 1] for index in range(len(array))]

def split_rolling(array, width):
	if width < 0:
		return split_rolling_out(array, -width)
	array = iterable(array)
	return [array[index : index + width] for index in range(len(array) - width + 1)]

def split_rolling_out(array, width):
	array = iterable(array)
	return [array[:index] + array[index + width:] for index in range(len(array) - width + 1)]

def split_suffix(array):
	array = iterable(array)
	return [array[index:] for index in range(len(array))]

def sss(compressed):
	from . import dictionary
	decompressed = ''
	integer = from_base([code_page.find(char) + 1 for char in compressed], 250)
	while integer:
		integer, mode = divmod(integer, 3)
		if mode == 0:
			integer, code = divmod(integer, 96)
			decompressed += code_page[code + 32]
		else:
			flag_swap = False
			flag_space = decompressed != ''
			if mode == 2:
				integer, flag = divmod(integer, 3)
				flag_swap = flag != 1
				flag_space ^= flag != 0
			integer, short = divmod(integer, 2)
			the_dictionary = (dictionary.long, dictionary.short)[short]
			integer, index = divmod(integer, len(the_dictionary))
			word = the_dictionary[index]
			if flag_swap:
				word = word[0].swapcase() + word[1:]
			if flag_space:
				word = ' ' + word
			decompressed += word
	return decompressed

def stringify(iterable, recurse = True):
	if type(iterable) != list:
		return iterable
	if len(iterable) == 1:
		return stringify(iterable[0])
	if str in map(type, iterable) and not list in map(type, iterable) or not iterable:
		return ''.join(map(str, iterable))
	iterable = [stringify(item) for item in iterable]
	return stringify(iterable, False) if recurse else iterable

def suffix(links, outmost_links, index):
	ret = [attrdict(arity = max(1, links[0].arity))]
	if len(links) == 1:
		ret[0].call = lambda x, y = None: [variadic_link(links[0], (t, y)) for t in split_suffix(x)]
	else:
		ret[0].call = lambda x, y = None: [variadic_link(links[0], (t, y)) for t in split_fixed(x, niladic_link(links[1]))]
	return ret

def symmetric_mod(number, half_divisor):
	modulus = number % (2 * half_divisor)
	return modulus - 2 * half_divisor * (modulus > half_divisor)

def tie(links, outmost_links, index):
	ret = [attrdict(arity= max(1, *arities(links)))]
	n = 2 if links[-1].arity else links[-1].call()
	def _make_tie():
		i = 0
		while True:
			yield links[i]
			i = (i + 1) % n
	cycle = _make_tie()
	ret[0].call = lambda x = None, y = None: variadic_link(next(cycle), (x, y))
	return ret

def time_format(bitfield):
	time_string = ':'.join(['%H'] * (bitfield & 4 > 0) + ['%M'] * (bitfield & 2 > 0) + ['%S'] * (bitfield & 1 > 0))
	return list(time.strftime(time_string))

def translate(mapping, array):
	array = iterable(array, make_copy = True)
	mapping = iterable(mapping, make_copy = True)
	while mapping:
		source = iterable(mapping.pop(0))
		destination = iterable(mapping.pop(0))
		for (index, item) in enumerate(array):
			if item in source:
				array[index] = destination[min(source.index(item), len(destination) - 1)]
	return array

def trim(trimmee, trimmer, left = False, right = False):
	lindex = 0
	rindex = len(trimmee)
	if left:
		while lindex < rindex and trimmee[lindex] in trimmer:
			lindex += 1
	if right:
		while lindex < rindex and trimmee[rindex - 1] in trimmer:
			rindex -= 1
	return trimmee[lindex:rindex]

def try_eval(string):
	try:
		return python_eval(string)
	except:
		return jellify(string, True)

def to_base(integer, base, bijective = False):
	if integer == 0:
		return [0] * (not bijective)
	if bijective:
		base = abs(base)
	if base == 0:
		return [integer]
	if base == -1:
		digits = [1, 0] * abs(integer)
		return digits[:-1] if integer > 0 else digits
	sign = -1 if integer < 0 and base > 0 else 1
	integer *= sign
	if base == 1:
		return [sign] * integer
	digits = []
	while integer:
		integer -= bijective
		integer, digit = divmod(integer, base)
		digit += bijective
		if digit < 0:
			integer += 1
			digit -= base
		digits.append(sign * digit)
	return digits[::-1]

def to_case(argument, lower = False, swap = False, title = False, upper = False):
	ret = []
	last_item = ' '
	for item in argument:
		if type(item) == str:
			if lower:
				ret.append(item.lower())
			elif swap:
				ret.append(item.swapcase())
			elif title:
				ret.append(item.lower() if type(last_item) == str and last_item in str_upper + str_lower else item.upper())
			elif upper:
				ret.append(item.upper())
		else:
			ret.append(item)
		last_item = item
	return ret

def to_exponents(integer):
	if integer == 1:
		return []
	pairs = sympy.ntheory.factor_.factorint(integer)
	exponents = []
	for prime in sympy.ntheory.generate.primerange(2, max(pairs) + 1):
		if prime in pairs:
			exponents.append(pairs[prime])
		else:
			exponents.append(0)
	return exponents

def to_factorial_base(integer):
	radix = 1
	digits = []
	while integer:
		integer, remainder = divmod(integer, radix)
		digits.append(remainder)
		radix += 1
	return digits[::-1] or [0]

def to_primorial_base(integer):
	placeIndex = 1
	integer, remainder = divmod(integer, 2)
	digits = [remainder]
	while integer:
		placeIndex += 1
		integer, remainder = divmod(integer, sympy.ntheory.generate.prime(placeIndex))
		digits.append(remainder)
	return digits[::-1]

def unicode_to_jelly(string):
	return ''.join(chr(code_page.find(char)) for char in str(string).replace('\n', '¶') if char in code_page)

def unique(array):
	array = iterable(array, make_digits = True)
	result = []
	for element in array:
		if not element in result:
			result.append(element)
	return result

def untruth_md(indices, shape = None, upper_level = []):
	if not shape:
		shape = [max(index_zipped) for index_zipped in zip(*indices)]
	upper_len = len(upper_level)
	if upper_len < len(shape) - 1:
		return [untruth_md(indices, shape = shape, upper_level = upper_level + [i + 1]) for i in range(shape[upper_len])]
	else:
		return [1 if (upper_level + [i + 1] in indices) else 0 for i in range(shape[-1])]

def variadic_chain(chain, args):
	args = list(filter(None.__ne__, args))
	larg_save = atoms['⁸'].call
	rarg_save = atoms['⁹'].call
	if len(args) == 0:
		return niladic_chain(chain)
	if len(args) == 1:
		atoms['⁸'].call = lambda literal = args[0]: literal
		ret = monadic_chain(chain, args[0])
	if len(args) == 2:
		atoms['⁸'].call = lambda literal = args[0]: literal
		atoms['⁹'].call = lambda literal = args[1]: literal
		ret = dyadic_chain(chain, args)
	atoms['⁸'].call = larg_save
	atoms['⁹'].call = rarg_save
	return ret

def variadic_link(link, args, flat = False, lflat = False, rflat = False):
	if link.arity < 0:
		args = list(filter(None.__ne__, args))
		link.arity = len(args)
	if link.arity == 0:
		return niladic_link(link)
	if link.arity == 1:
		return monadic_link(link, args[0], flat)
	if link.arity == 2:
		return dyadic_link(link, args, lflat, rflat)

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

def windowed_exists(needle, haystack):
	haystack = iterable(haystack, make_digits = True)
	needle = iterable(needle, make_digits = True)
	needle_length = len(needle)
	for index in range(len(haystack)):
		if haystack[index : index + needle_length] == needle:
			return 1
	return 0

def windowed_index_of(haystack, needle):
	haystack = iterable(haystack, make_digits = True)
	needle = iterable(needle, make_digits = True)
	needle_length = len(needle)
	for index in range(len(haystack)):
		if haystack[index : index + needle_length] == needle:
			return 1 + index
	return 0

def windowed_sublists(array):
	array = iterable(array, make_range = True)
	return [sublist for width in range(1, len(array) + 1) for sublist in split_rolling(array, width)]

def output(argument, end = '', transform = stringify):
	if locale.getdefaultlocale()[1][0:3] == 'UTF':
		print(transform(argument), end = end)
	else:
		print(unicode_to_jelly(transform(argument)), end = unicode_to_jelly(end))
	sys.stdout.flush()
	return argument

def zip_ragged(array):
	return jellify(map(lambda t: filter(None.__ne__, t), itertools.zip_longest(*map(iterable, array))))

atoms = {
	'³': attrdict(
		arity = 0,
		call = lambda: 100
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
	'⁸': attrdict(
		arity = 0,
		call = lambda: []
	),
	'⁹': attrdict(
		arity = 0,
		call = lambda: 256
	),
	'A': attrdict(
		arity = 1,
		ldepth = 0,
		call = abs
	),
	'Ȧ': attrdict(
		arity = 1,
		call = lambda z: int(iterable(z) > [] and all(flatten(z)))
	),
	'Ạ': attrdict(
		arity = 1,
		call = lambda z: int(all(iterable(z)))
	),
	'a': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: x and y
	),
	'ȧ': attrdict(
		arity = 2,
		call = lambda x, y: x and y
	),
	'ạ': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: abs(x - y)
	),
	'B': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: to_base(z, 2)
	),
	'Ḅ': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: from_base(z, 2)
	),
	'Ḃ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: z % 2
	),
	'b': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: to_base(x, y)
	),
	'ḅ': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 0,
		call = lambda x, y: from_base(x, y)
	),
	'ḃ': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: to_base(x, y, bijective = True)
	),
	'C': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: 1 - z
	),
	'Ċ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.ceil, lambda t: t.imag, identity), z)
	),
	'c': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = nCr
	),
	'ċ': attrdict(
		arity = 2,
		call = lambda x, y: iterable(x).count(y)
	),
	'ƈ': attrdict(
		arity = 0,
		call = lambda: sys.stdin.read(1) or []
	),
	'D': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: to_base(z, 10)
	),
	'Ḍ': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: from_base(z, 10)
	),
	'Ḋ': attrdict(
		arity = 1,
		call = lambda z: iterable(z, make_range = True)[1:]
	),
	'd': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: list(divmod(x, y))
	),
	'ḍ': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: int(y % x == 0 if x else y == 0)
	),
	'ḋ': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 1,
		call = lambda x, y: dot_product(x, y, truncate = True)
	),
	'E': attrdict(
		arity = 1,
		call = equal
	),
	'Ẹ': attrdict(
		arity = 1,
		call = lambda z: int(any(iterable(z)))
	),
	'Ė': attrdict(
		arity = 1,
		call = lambda z: [[t + 1, u] for t, u in enumerate(iterable(z))]
	),
	'e': attrdict(
		arity = 2,
		call = lambda x, y: int(x in iterable(y))
	),
	'ẹ': attrdict(
		arity = 2,
		call = lambda x, y: [t + 1 for t, u in enumerate(iterable(x, make_digits = True)) if u == y]
	),
	'F': attrdict(
		arity = 1,
		call = flatten
	),
	'Ḟ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.floor, lambda t: t.real, identity), z)
	),
	'f': attrdict(
		arity = 2,
		call = filter_array
	),
	'ḟ': attrdict(
		arity = 2,
		call = lambda x, y: filter_array(x, y, False)
	),
	'G': attrdict(
		arity = 1,
		call = grid
	),
	'Ġ': attrdict(
		arity = 1,
		call = group
	),
	'g': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = math.gcd
	),
	'Ɠ': attrdict(
		arity = 0,
		call = lambda: python_eval(input())
	),
	'ɠ': attrdict(
		arity = 0,
		call = lambda: jellify(input(), dirty = True)
	),
	'H': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: div(z, 2)
	),
	'Ḥ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: z * 2
	),
	'Ḣ': attrdict(
		arity = 1,
		call = lambda z: iterable(z).pop(0) if iterable(z) else 0
	),
	'ḣ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: iterable(x)[:y]
	),
	'I': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: [z[i] - z[i - 1] for i in range(1, len(z))]
	),
	'İ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: div(1, z)
	),
	'Ị': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(abs(z) <= 1)
	),
	'J': attrdict(
		arity = 1,
		call = lambda z: list(range(1, len(iterable(z)) + 1))
	),
	'i': attrdict(
		arity = 2,
		call = index_of
	),
	'ḥ': attrdict(
		arity = 2,
		call = jelly_hash
	),
	'ị': attrdict(
		arity = 2,
		ldepth = 0,
		call = at_index
	),
	'j': attrdict(
		arity = 2,
		call = join
	),
	'K': attrdict(
		arity = 1,
		call = lambda z: join(z, ' ')
	),
	'Ḳ': attrdict(
		arity = 1,
		call = lambda z: jellify(split_at(iterable(z), ' '))
	),
	'k': attrdict(
		arity = 2,
		call = lambda x, y: partition_at(x, y, border = 2)
	),
	'L': attrdict(
		arity = 1,
		call = lambda z: len(iterable(z))
	),
	'Ḷ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: list(range(int(z)))
	),
	'l': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: overload((math.log, cmath.log), x, y)
	),
	'ḷ': attrdict(
		arity = 2,
		call = lambda x, y: x
	),
	'M': attrdict(
		arity = 1,
		call = maximal_indices
	),
	'Ṃ': attrdict(
		arity = 1,
		call = lambda z: min(iterable(z)) if iterable(z) else 0
	),
	'Ṁ': attrdict(
		arity = 1,
		call = lambda z: max(iterable(z)) if iterable(z) else 0
	),
	'm': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: iterable(x)[::y] if y else iterable(x) + iterable(x)[::-1]
	),
	'ṁ': attrdict(
		arity = 2,
		call = lambda x, y: mold(iterable(x, make_copy = True), iterable(y, make_copy = True, make_range = True))
	),
	'ṃ': attrdict(
		arity = 2,
		ldepth = 0,
		call = base_decompression
	),
	'N': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: -z
	),
	'Ṅ': attrdict(
		arity = 1,
		call = lambda z: output(z, end = '\n')
	),
	'Ṇ': attrdict(
		arity = 1,
		call = lambda z: int(not(z))
	),
	'n': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: int(x != y)
	),
	'O': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: ord(z) if type(z) == str else z
	),
	'Ọ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: chr(int(z)) if type(z) != str else z
	),
	'Ȯ': attrdict(
		arity = 1,
		call = output
	),
	'o': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: x or y
	),
	'ọ': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = order
	),
	'ȯ': attrdict(
		arity = 2,
		call = lambda x, y: x or y
	),
	'P': attrdict(
		arity = 1,
		call = lambda z: functools.reduce(lambda x, y: dyadic_link(atoms['×'], (x, y)), z, 1) if type(z) == list else z
	),
	'Ṗ': attrdict(
		arity = 1,
		call = lambda z: iterable(z, make_range = True)[:-1]
	),
	'p': attrdict(
		arity = 2,
		call = lambda x, y: jellify(itertools.product(iterable(x, make_range = True), iterable(y, make_range = True)))
	),
	'ṗ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: jellify(itertools.product(*([iterable(x, make_range = True)] * y)))
	),
	'Q': attrdict(
		arity = 1,
		call = unique
	),
	'R': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: list(range(1, int(z) + 1))
	),
	'Ṛ': attrdict(
		arity = 1,
		call = lambda z: iterable(z, make_digits = True)[::-1]
	),
	'Ṙ': attrdict(
		arity = 1,
		call = lambda z: output(z, transform = jelly_uneval)
	),
	'r': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: list(range(int(x), int(y) + 1) or range(int(x), int(y) - 1, -1))
	),
	'ṙ': attrdict(
		arity = 2,
		rdepth = 0,
		call = rotate_left
	),
	'ṛ': attrdict(
		arity = 2,
		call = lambda x, y: y
	),
	'S': attrdict(
		arity = 1,
		call = lambda z: functools.reduce(lambda x, y: dyadic_link(atoms['+'], (x, y)), z, 0) if type(z) == list else z
	),
	'Ṡ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((lambda t: (t > 0) - (t < 0), lambda t: t.conjugate()), z)
	),
	'Ṣ': attrdict(
		arity = 1,
		call = lambda z: sorted(iterable(z, make_digits = True))
	),
	's': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: split_fixed(iterable(x, make_range = True), y)
	),
	'ṡ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: split_rolling(iterable(x, make_range = True), y)
	),
	'ṣ': attrdict(
		arity = 2,
		call = lambda x, y: jellify(split_at(iterable(x, make_digits = True), y))
	),
	'T': attrdict(
		arity = 1,
		call = lambda z: [u + 1 for u, v in enumerate(z) if v]
	),
	'Ṭ': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: [int(t + 1 in iterable(z)) for t in range(max(iterable(z) or [0]))]
	),
	'Ṫ': attrdict(
		arity = 1,
		call = lambda z: iterable(z).pop() if iterable(z) else 0
	),
	't': attrdict(
		arity = 2,
		call = lambda x, y: trim(x, iterable(y), left = True, right = True)
	),
	'ṫ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: iterable(x)[y - 1 :]
	),
	'ṭ': attrdict(
		arity = 2,
		call = lambda x, y: iterable(y) + [x]
	),
	'U': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: z[::-1]
	),
	'Ụ': attrdict(
		arity = 1,
		call = lambda z: sorted(range(1, len(iterable(z)) + 1), key = lambda t: iterable(z)[t - 1])
	),
	'V': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: jelly_eval(''.join(map(str, z)), [])
	),
	'Ṿ': attrdict(
		arity = 1,
		call = lambda z: jellify(jelly_uneval(z))
	),
	'v': attrdict(
		arity = 2,
		ldepth = 1,
		call = lambda x, y: jelly_eval(''.join(map(str, x)), [y])
	),
	'W': attrdict(
		arity = 1,
		call = lambda z: [z]
	),
	'Ẇ': attrdict(
		arity = 1,
		call = windowed_sublists
	),
	'Ẉ': attrdict(
		arity = 1,
		call = lambda z: [len(iterable(t, make_digits = True)) for t in iterable(z, make_range = True)]
	),
	'w': attrdict(
		arity = 2,
		call = windowed_index_of
	),
	'ẇ': attrdict(
		arity = 2,
		call = windowed_exists
	),
	'X': attrdict(
		arity = 1,
		call = random_int
	),
	'Ẋ': attrdict(
		arity = 1,
		call = shuffle
	),
	'x': attrdict(
		arity = 2,
		ldepth = 1,
		call = lambda x, y: rld(zip(x, y if depth(y) else [y] * len(x)))
	),
	'ẋ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: iterable(x) * int(y)
	),
	'Y': attrdict(
		arity = 1,
		call = lambda z: join(z, '\n')
	),
	'Ỵ': attrdict(
		arity = 1,
		call = lambda z: jellify(split_at(iterable(z), '\n'))
	),
	'Ẏ': attrdict(
		arity = 1,
		call = lambda z: sum(map(iterable, iterable(z)), [])
	),
	'y': attrdict(
		arity = 2,
		call = translate
	),
	'Z': attrdict(
		arity = 1,
		call = zip_ragged
	),
	'Ż': attrdict(
		arity = 1,
		call = lambda z: [0] + iterable(z, make_range = True)
	),
	'z': attrdict(
		arity = 2,
		call = lambda x, y: jellify(itertools.zip_longest(*map(iterable, x), fillvalue = y))
	),
	'ż': attrdict(
		arity = 2,
		call = lambda x, y: zip_ragged([x, y])
	),
	'§': attrdict(
		arity = 1,
		ldepth = 1,
		call = sum
	),
	'Ä': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: list(itertools.accumulate(z))
	),
	'!': attrdict(
		arity = 1,
		ldepth = 0,
		call = Pi
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
		call = lambda x, y: div(x, y, floor = True)
	),
	',': attrdict(
		arity = 2,
		call = lambda x, y: [x, y]
	),
	';': attrdict(
		arity = 2,
		call = lambda x, y: iterable(x) + iterable(y)
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
		call = div
	),
	'%': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = modulus
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
		conv = conv_dyadic_integer,
		call = operator.and_
	),
	'^': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		conv = conv_dyadic_integer,
		call = operator.xor
	),
	'|': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		conv = conv_dyadic_integer,
		call = operator.or_
	),
	'~': attrdict(
		arity = 1,
		ldepth = 0,
		conv = conv_monadic_integer,
		call = lambda z: ~z
	),
	'²': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: z ** 2
	),
	'½': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.sqrt, cmath.sqrt), z)
	),
	'°': attrdict(
		arity = 1,
		ldepth = 0,
		call = math.radians
	),
	'¬': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(not z)
	),
	'‘': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: z + 1
	),
	'’': attrdict(
		arity = 1,
		ldepth = 0,
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
	'⁼': attrdict(
		arity = 2,
		call = lambda x, y: int(x == y)
	),
	'⁻': attrdict(
		arity = 2,
		call = lambda x, y: int(x != y)
	),
	'®': attrdict(
		arity = 0,
		call = lambda: 0
	),
	'¹': attrdict(
		arity = 1,
		call = identity
	),
	'ÆA': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.acos, cmath.acos), z)
	),
	'ÆẠ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.cos, cmath.cos), z)
	),
	'ÆC': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.generate.primepi(z)
	),
	'ÆĊ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.functions.combinatorial.numbers.catalan(z))
	),
	'Æc': attrdict(
		arity = 1,
		ldepth = 0,
		call = carmichael
	),
	'ÆD': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.factor_.divisors(z)
	),
	'ÆḌ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.factor_.divisors(z)[:-1]
	),
	'Æd': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.ntheory.factor_.divisor_count(z))
	),
	'Æḍ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.ntheory.factor_.divisor_count(z) - 1)
	),
	'ÆḊ': attrdict(
		arity = 1,
		ldepth = 2,
		call = determinant
	),
	'ÆE': attrdict(
		arity = 1,
		ldepth = 0,
		call = to_exponents
	),
	'ÆẸ': attrdict(
		arity = 1,
		ldepth = 1,
		call = from_exponents
	),
	'ÆF': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: [[x, y] for x, y in sorted(sympy.ntheory.factor_.factorint(z).items())]
	),
	'Æe': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.exp, cmath.exp), z)
	),
	'Æf': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: rld(sorted(sympy.ntheory.factor_.factorint(z).items()))
	),
	'ÆḞ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.functions.combinatorial.numbers.fibonacci(z))
	),
	'Æi': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: [z.real, z.imag]
	),
	'Æị': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: complex(*z[:2])
	),
	'ÆĿ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.functions.combinatorial.numbers.lucas(z))
	),
	'Æl': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.log, cmath.log), z)
	),
	'Æm': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: div(sum(z), len(z))
	),
	'Æṁ': attrdict(
		arity = 1,
		ldepth = 1,
		call = median
	),
	'Æṃ': attrdict(
		arity = 1,
		ldepth = 1,
		call = mode
	),
	'ÆN': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.generate.prime(z)
	),
	'Æn': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.generate.nextprime(z)
	),
	'ÆP': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.primetest.isprime(z))
	),
	'Æp': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.generate.prevprime(z)
	),
	'ÆR': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: list(sympy.ntheory.generate.primerange(2, z + 1))
	),
	'Ær': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: jellify(from_base(z[::-1], sympy.poly('x')).all_roots())
	),
	'Æṛ': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: jellify(sympy.prod(map(sympy.poly('x').__sub__, z)).all_coeffs()[::-1])
	),
	'ÆT': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.tan, cmath.tan), z)
	),
	'ÆṬ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.atan, cmath.atan), z)
	),
	'ÆṪ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sympy.ntheory.factor_.totient(z) if z > 0 else 0
	),
	'Æṭ': attrdict(
		arity = 1,
		ldepth = 2,
		call = lambda z: sum(sum(r[i : i+1]) for i, r in enumerate(z))
	),
	'ÆS': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.sin, cmath.sin), z)
	),
	'ÆṢ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.asin, cmath.asin), z)
	),
	'Æs': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.ntheory.factor_.divisor_sigma(z))
	),
	'Æṣ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.ntheory.factor_.divisor_sigma(z) - z)
	),
	'Æv': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: len(sympy.ntheory.factor_.factorint(z))
	),
	'Æ²': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(isqrt(z) ** 2 == z)
	),
	'Æ½': attrdict(
		arity = 1,
		ldepth = 0,
		call = isqrt
	),
	'Æ°': attrdict(
		arity = 1,
		ldepth = 0,
		call = math.degrees
	),
	'Æ!': attrdict(
		arity = 1,
		ldepth = 0,
		call = to_factorial_base
	),
	'Æ¡': attrdict(
		arity = 1,
		ldepth = 1,
		call = from_factorial_base
	),
	'Æ?': attrdict(
		arity = 1,
		ldepth = 0,
		call = to_primorial_base
	),
	'Æ¿': attrdict(
		arity = 1,
		ldepth = 1,
		call = from_primorial_base
	),
	'Œ!': attrdict(
		arity = 1,
		call = lambda z: jellify(itertools.permutations(iterable(z, make_range = True)))
	),
	'Œ?': attrdict(
		arity = 1,
		ldepth = 0,
		call = pemutation_at_index
	),
	'Œ¿': attrdict(
		arity = 1,
		call = permutation_index
	),
	'ŒB': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: bounce(z)
	),
	'ŒḄ': attrdict(
		arity = 1,
		call = lambda z: bounce(iterable(z, make_range = True))
	),
	'ŒḂ': attrdict(
		arity = 1,
		call = is_palindrome
	),
	'Œb': attrdict(
		arity = 1,
		call = lambda z: partitions(z) if z else [[]]
	),
	'Œc': attrdict(
		arity = 1,
		rdepth = 0,
		call = lambda z: jellify(itertools.combinations(iterable(z, make_range = True), 2))
	),
	'Œċ': attrdict(
		arity = 1,
		rdepth = 0,
		call = lambda z: jellify(itertools.combinations_with_replacement(iterable(z, make_range = True), 2))
	),
	'ŒD': attrdict(
		arity = 1,
		ldepth = 2,
		call = diagonals
	),
	'ŒḌ': attrdict(
		arity = 1,
		ldepth = 2,
		call = from_diagonals
	),
	'ŒḊ': attrdict(
		arity = 1,
		call = depth
	),
	'Œd': attrdict(
		arity = 1,
		ldepth = 2,
		call = lambda z: diagonals([r[::-1] for r in z])
	),
	'Œḍ': attrdict(
		arity = 1,
		ldepth = 2,
		call = lambda z: [r[::-1] for r in from_diagonals(z)]
	),
	'ŒĖ': attrdict(
		arity = 1,
		call = lambda z: list(enumerate_md(z))
	),
	'Œe': attrdict(
		arity = 1,
		call = lambda z: [t for t in iterable(z, make_range = True)[1::2]]
	),
	'ŒG': attrdict(
		arity = 1,
		ldepth = 1,
		call = get_request
	),
	'ŒĠ': attrdict(
		arity = 1,
		call = group_md
	),
	'Œg': attrdict(
		arity = 1,
		ldepth = 1,
		call = group_equal
	),
	'ŒH': attrdict(
		arity = 1,
		call = lambda z: split_evenly(iterable(z, make_range = True), 2)
	),
	'œị': attrdict(
		arity = 2,
		ldepth = 1,
		call = at_index_ndim
	),
	'ŒJ': attrdict(
		arity = 1,
		call = indices_md
	),
	'Œl': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: to_case(z, lower = True)
	),
	'ŒM': attrdict(
		arity = 1,
		call = maximal_indices_md
	),
	'Œo': attrdict(
		arity = 1,
		call = lambda z: [t for t in iterable(z, make_range = True)[::2]]
	),
	'ŒP': attrdict(
		arity = 1,
		call = powerset
	),
	'ŒṖ': attrdict(
		arity = 1,
		call = partitions
	),
	'Œṗ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: sorted(integer_partitions(z), key = len, reverse = True)
	),
	'Œp': attrdict(
		arity = 1,
		call = lambda z: jellify(itertools.product(*[iterable(t, make_range = True) for t in z]))
	),
	'ŒQ': attrdict(
		arity = 1,
		call = distinct_sieve
	),
	'ŒR': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: list(range(-abs(int(z)), abs(int(z)) + 1))
	),
	'ŒṘ': attrdict(
		arity = 1,
		call = lambda z: jellify(repr(z))
	),
	'Œr': attrdict(
		arity = 1,
		ldepth = 1,
		call = rle
	),
	'Œṙ': attrdict(
		arity = 1,
		ldepth = 2,
		call = rld
	),
	'Œs': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: to_case(z, swap = True)
	),
	'ŒT': attrdict(
		arity = 1,
		call = time_format
	),
	'ŒṬ': attrdict(
		arity = 1,
		ldepth = 2,
		call = untruth_md
	),
	'ŒṪ': attrdict(
		arity = 1,
		call = lambda z: [t for t, u in enumerate_md(iterable(z)) if u]
	),
	'Œt': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: to_case(z, title = True)
	),
	'ŒV': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: python_eval(''.join(map(str, z)))
	),
	'ŒỤ': attrdict(
		arity = 1,
		call = lambda z: sorted(indices_md(iterable(z)), key = lambda t: at_index_ndim(t, iterable(z)))
	),
	'Œu': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: to_case(z, upper = True)
	),
	'Œœ': attrdict(
		arity = 1,
		call = odd_even
	),
	'Œɠ': attrdict(
		arity = 1,
		call = group_lengths
	),
	'œ?': attrdict(
		arity = 2,
		ldepth = 0,
		call = pemutation_at_index
	),
	'œ¿': attrdict(
		arity = 2,
		call = lambda x, y: permutation_index([y.index(value) for value in x])
	),
	'æ.': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 1,
		call = dot_product
	),
	'æ%': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = symmetric_mod
	),
	'æ*': attrdict(
		arity = 2,
		ldepth = 2,
		rdepth = 0,
		call = lambda x, y: matrix_to_list((sympy.Matrix(x) ** y))
	),
	'æ×': attrdict(
		arity = 2,
		ldepth = 2,
		rdepth = 2,
		call = lambda x, y: matrix_to_list((sympy.Matrix(x) * sympy.Matrix(y)))
	),
	'æA': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = math.atan2
	),
	'æR': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = primerange
	),
	'æC': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 0,
		call = convolve_power
	),
	'æc': attrdict(
		arity = 2,
		ldepth = 1,
		rdepth = 1,
		call = convolve
	),
	'æċ': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: from_base([1] + [0] * len(to_base(x, y)), y)
	),
	'æḟ': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: from_base([1] + [0] * (len(to_base(x, y)) - 1), y)
	),
	'æi': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = modinv
	),
	'æị': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = complex
	),
	'æl': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lcm
	),
	'ær': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = round
	),
	'æp': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: float('%%.%dg'%y%x)
	),
	'æ«': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = shift_left
	),
	'æ»': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = shift_right
	),
	'œ!': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: jellify(itertools.permutations(iterable(x, make_range = True), y))
	),
	'œc': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: jellify(itertools.combinations(iterable(x, make_range = True), y))
	),
	'œċ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: jellify(itertools.combinations_with_replacement(iterable(x, make_range = True), y))
	),
	'œẹ': attrdict(
		arity = 2,
		call = lambda x, y: [t for t, u in enumerate_md(iterable(x)) if u == y]
	),
	'œi': attrdict(
		arity = 2,
		call = index_of_md
	),
	'œl': attrdict(
		arity = 2,
		call = lambda x, y: trim(x, iterable(y), left = True)
	),
	'œP': attrdict(
		arity = 2,
		call = lambda x, y: partition_at([int(t + 1 in iterable(x)) for t in range(max(iterable(x) or [0]))], y, border = 0)
	),
	'œṖ': attrdict(
		arity = 2,
		call = lambda x, y: partition_at([int(t + 1 in iterable(x)) for t in range(max(iterable(x) or [0]))], y)
	),
	'œp': attrdict(
		arity = 2,
		call = lambda x, y: partition_at(x, y, border = 0)
	),
	'œṗ': attrdict(
		arity = 2,
		call = partition_at
	),
	'œr': attrdict(
		arity = 2,
		call = lambda x, y: trim(x, iterable(y), right = True)
	),
	'œS': attrdict(
		arity = 2,
		call = lambda x, y: time.sleep(overload((float, bool), y)) or x
	),
	'œs': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: split_evenly(iterable(x, make_range = True), y)
	),
	'œṡ': attrdict(
		arity = 2,
		rdepth = 0,
		call = split_once
	),
	'œṣ': attrdict(
		arity = 2,
		call = lambda x, y: jellify(split_around(iterable(x, make_digits = True), iterable(y)))
	),
	'œ&': attrdict(
		arity = 2,
		call = multiset_intersect
	),
	'œ-': attrdict(
		arity = 2,
		call = multiset_difference
	),
	'œ^': attrdict(
		arity = 2,
		call = multiset_symdif
	),
	'œ|': attrdict(
		arity = 2,
		call = multiset_union
	),
	'Ø0': attrdict(
		arity = 0,
		call = lambda: [0, 0]
	),
	'Ø1': attrdict(
		arity = 0,
		call = lambda: [1, 1]
	),
	'Ø2': attrdict(
		arity = 0,
		call = lambda: [2, 2]
	),
	'Ø.': attrdict(
		arity = 0,
		call = lambda: [0, 1]
	),
	'Ø½': attrdict(
		arity = 0,
		call = lambda: [1, 2]
	),
	'Ø+': attrdict(
		arity = 0,
		call = lambda: [1, -1]
	),
	'Ø-': attrdict(
		arity = 0,
		call = lambda: [-1, 1]
	),
	'Ø(': attrdict(
		arity = 0,
		call = lambda: list('()')
	),
	'Ø<': attrdict(
		arity = 0,
		call = lambda: list('<>')
	),
	'Ø[': attrdict(
		arity = 0,
		call = lambda: list('[]')
	),
	'Ø{': attrdict(
		arity = 0,
		call = lambda: list('{}')
	),
	'Ø^': attrdict(
		arity = 0,
		call = lambda: list('/\\')
	),
	'Ø⁵': attrdict(
		arity = 0,
		call = lambda: 250
	),
	'Ø⁷': attrdict(
		arity = 0,
		call = lambda: 128
	),
	'Ø°': attrdict(
		arity = 0,
		call = lambda: 360
	),
	'Ø%': attrdict(
		arity = 0,
		call = lambda: 2 ** 32
	),
	'ØA': attrdict(
		arity = 0,
		call = lambda: list(str_upper)
	),
	'ØẠ': attrdict(
		arity = 0,
		call = lambda: list(str_upper + str_lower)
	),
	'ØB': attrdict(
		arity = 0,
		call = lambda: list(str_digit + str_upper + str_lower)
	),
	'ØḄ': attrdict(
		arity = 0,
		call = lambda: list('bcdfghjklmnpqrstvwxyz')
	),
	'ØḂ': attrdict(
		arity = 0,
		call = lambda: list('BCDFGHJKLMNPQRSTVWXYZ')
	),
	'ØC': attrdict(
		arity = 0,
		call = lambda: list('BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz')
	),
	'ØD': attrdict(
		arity = 0,
		call = lambda: list(str_digit)
	),
	'ØH': attrdict(
		arity = 0,
		call = lambda: list(str_digit + 'ABCDEF')
	),
	'ØJ': attrdict(
		arity = 0,
		call = lambda: list(code_page)
	),
	'ØP': attrdict(
		arity = 0,
		call = lambda: math.pi
	),
	'ØṖ': attrdict(
		arity = 0,
		call = lambda: list(map(chr, range(32, 127)))
	),
	'ØQ': attrdict(
		arity = 0,
		call = lambda: [list('QWERTYUIOP'), list('ASDFGHJKL'), list('ZXCVBNM')]
	),
	'ØV': attrdict(
		arity = 0,
		call = lambda: list('ṘV')
	),
	'ØW': attrdict(
		arity = 0,
		call = lambda: list(str_upper + str_lower + str_digit + '_')
	),
	'ØY': attrdict(
		arity = 0,
		call = lambda: list('BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz')
	),
	'ØỴ': attrdict(
		arity = 0,
		call = lambda: list('bcdfghjklmnpqrstvwxz')
	),
	'ØẎ': attrdict(
		arity = 0,
		call = lambda: list('BCDFGHJKLMNPQRSTVWXZ')
	),
	'Øa': attrdict(
		arity = 0,
		call = lambda: list(str_lower)
	),
	'Øb': attrdict(
		arity = 0,
		call = lambda: list(str_upper + str_lower + str_digit + '+/')
	),
	'Øc': attrdict(
		arity = 0,
		call = lambda: list('AEIOUaeiou')
	),
	'Øe': attrdict(
		arity = 0,
		call = lambda: math.e
	),
	'Øẹ': attrdict(
		arity = 0,
		call = lambda: list('aeiou')
	),
	'Øė': attrdict(
		arity = 0,
		call = lambda: list('AEIOU')
	),
	'Øh': attrdict(
		arity = 0,
		call = lambda: list(str_digit + 'abcdef')
	),
	'Øp': attrdict(
		arity = 0,
		call = lambda: (1 + math.sqrt(5)) / 2
	),
	'Øq': attrdict(
		arity = 0,
		call = lambda: [list('qwertyuiop'), list('asdfghjkl'), list('zxcvbnm')]
	),
	'Øv': attrdict(
		arity = 0,
		call = lambda: list('Ṙv')
	),
	'Øy': attrdict(
		arity = 0,
		call = lambda: list('AEIOUYaeiouy')
	),
	'Øỵ': attrdict(
		arity = 0,
		call = lambda: list('aeiouy')
	),
	'Øẏ': attrdict(
		arity = 0,
		call = lambda: list('AEIOUY')
	)
}

quicks = {
	'©': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x = None, y = None: copy_to(atoms['®'], variadic_link(links[0], (x, y)))
		)]
	),
	'ß': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [create_chain(outmost_links[index])]
	),
	'¢': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [create_chain(outmost_links[index - 1], 0)]
	),
	'Ç': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [create_chain(outmost_links[index - 1], 1)]
	),
	'ç': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [create_chain(outmost_links[index - 1], 2)]
	),
	'Ñ': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [create_chain(outmost_links[(index + 1) % len(outmost_links)], 1)]
	),
	'ñ': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [create_chain(outmost_links[(index + 1) % len(outmost_links)], 2)]
	),
	'¦': attrdict(
		condition = lambda links: len(links) == 2,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max_arity(links + [atoms['¹']]),
			call = lambda x, y = None: sparse(links[0], (x, y), links[1])
		)]
	),
	'¡': attrdict(
		condition = lambda links: len(links) == 2,
		quicklink = lambda links, outmost_links, index: ([links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []) + [attrdict(
			arity = max_arity(links),
			call = lambda x = None, y = None: ntimes(links, (x, y))
		)]
	),
	'¿': attrdict(
		condition = lambda links: len(links) == 2,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(link.arity for link in links),
			call = lambda x = None, y = None: while_loop(links[0], links[1], (x, y))
		)]
	),
	'/': attrdict(
		condition = lambda links: links and links[0].arity,
		quicklink = reduce
	),
	'\\': attrdict(
		condition = lambda links: links and links[0].arity,
		quicklink = reduce_cumulative
	),
	'Ƒ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(1, links[0].arity),
			call = lambda x, y = None: int(x == variadic_link(links[0], (x, y)))
		)]
	),
	'ƒ': attrdict(
		condition = lambda links: links and links[0].arity,
		quicklink = foldl
	),
	'Ɲ': attrdict(
		condition = lambda links: links and not leading_nilad(links),
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 1,
			call = lambda z: neighbors(links, z)
		)]
	),
	'Ƥ': attrdict(
		condition = lambda links: links and links[0].arity,
		quicklink = prefix
	),
	'ÐƤ': attrdict(
		condition = lambda links: links and links[0].arity,
		quicklink = suffix
	),
	'ƙ': attrdict(
		condition = lambda links: links and links[0].arity,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 2,
			call = lambda x, y: [monadic_link(links[0], g) for g in split_key(iterable(x, make_digits = True), iterable(y, make_digits = True))]
		)]
	),
	'ɼ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(0, links[0].arity - 1),
			call = lambda z = None: copy_to(atoms['®'], variadic_link(links[0], (niladic_link(atoms['®']), z)))
		)]
	),
	'Ƭ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x = None, y = None: loop_until_loop(links[0], (x, y), return_all = True, vary_rarg = False)
		)]
	),
	'ƭ': attrdict(
		condition = lambda links: links and (
			(links[-1].arity == 0 and len(links) - 1 == links[-1].call()) or
			(links[-1].arity and len(links) == 2)),
		quicklink = tie
	),
	'¤': quickchain(0, 2),
	'$': quickchain(1, 2),
	'Ɗ': quickchain(1, 3),
	'Ʋ': quickchain(1, 4),
	'¥': quickchain(2, 2),
	'ɗ': quickchain(2, 3),
	'ʋ': quickchain(2, 4),
	'#': attrdict(
		condition = lambda links: len(links) == 2,
		quicklink = lambda links, outmost_links, index: ([links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []) + [attrdict(
			arity = max_arity(links),
			call = lambda x = None, y = None: nfind(links, (x, y))
		)]
	),
	'?': attrdict(
		condition = lambda links: len(links) == 3,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(link.arity for link in links),
			call = lambda x = None, y = None: variadic_link(links[0], (x, y)) if variadic_link(links[2], (x, y)) else variadic_link(links[1], (x, y))
		)]
	),
	'`': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 1,
			call = lambda z: dyadic_link(links[0], (z, z))
		)]
	),
	'⁺': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: links * 2
	),
	'Ð¡': attrdict(
		condition = lambda links: len(links) == 2,
		quicklink = lambda links, outmost_links, index: ([links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []) + [attrdict(
			arity = max(link.arity for link in links),
			call = lambda x = None, y = None: ntimes(links, (x, y), cumulative = True)
		)]
	),
	'Ð¿': attrdict(
		condition = lambda links: len(links) == 2,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(link.arity for link in links),
			call = lambda x = None, y = None: while_loop(links[0], links[1], (x, y), cumulative = True)
		)]
	),
	'Ðe': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(1, links[0].arity),
			call = lambda x, y = None: sparse(links[0], (x, y), range(2, len(x) + 2, 2), indices_literal = True)
		)]
	),
	'Ðf': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x, y = None: list(filter(lambda t: variadic_link(links[0], (t, y)), iterable(x, make_range = True)))
		)]
	),
	'Ðḟ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x, y = None: list(itertools.filterfalse(lambda t: variadic_link(links[0], (t, y)), iterable(x, make_range = True)))
		)]
	),
	'ÐL': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x = None, y = None: loop_until_loop(links[0], (x, y))
		)]
	),
	'ÐĿ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x = None, y = None: loop_until_loop(links[0], (x, y), return_all = True)
		)]
	),
	'ÐḶ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x = None, y = None: loop_until_loop(links[0], (x, y), return_loop = True)
		)]
	),
	'ÐṂ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x, y = None: extremes(min, links[0], (x, y))
		)]
	),
	'ÐṀ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x, y = None: extremes(max, links[0], (x, y))
		)]
	),
	'Ðo': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = max(1, links[0].arity),
			call = lambda x, y = None: sparse(links[0], (x, y), range(1, len(x) + 1, 2), indices_literal = True)
		)]
	),
}

hypers = {
	'"': lambda link, none = None: attrdict(
		arity = 2,
		call = lambda x, y: [dyadic_link(link, (u, v)) for u, v in zip(iterable(x), iterable(y))] + iterable(x)[len(iterable(y)) :] + iterable(y)[len(iterable(x)) :]
	),
	"'": lambda link, none = None: attrdict(
		arity = link.arity,
		call = lambda x = None, y = None: variadic_link(link, (x, y), flat = True, lflat = True, rflat = True)
	),
	'@': lambda link, none = None: attrdict(
		arity = 2,
		call = lambda x, y: dyadic_link(link, (y, x))
	),
	'{': lambda link, none = None: attrdict(
		arity = 2,
		call = lambda x, y: monadic_link(link, x)
	),
	'}': lambda link, none = None: attrdict(
		arity = 2,
		call = lambda x, y: monadic_link(link, y)
	),
	'€': lambda link, none = None: attrdict(
		arity = max(1, link.arity),
		call = lambda x, y = None: [variadic_link(link, (t, y)) for t in iterable(x, make_range = True)]
	),
	'Þ': lambda link, none = None: attrdict(
		arity = link.arity,
		call = lambda x, y = None: sorted(iterable(x, make_range = True), key=lambda t: variadic_link(link, (t, y)))
	),
	'þ': lambda link, none = None: attrdict(
		arity = 2,
		call = lambda x, y: [[dyadic_link(link, (u, v)) for u in iterable(x, make_range = True)] for v in iterable(y, make_range = True)]
	),
	'Ð€': lambda link, none = None: attrdict(
		arity = max(1, link.arity),
		call = lambda x, y = None: [variadic_link(link, (x, t)) for t in iterable(y, make_range = True)]
	),
	'£': lambda index, links: attrdict(
		arity = index.arity,
		call = lambda x = None, y = None: niladic_chain(links[(variadic_link(index, (x, y)) - 1) % (len(links) - 1)])
	),
	'Ŀ': lambda index, links: attrdict(
		arity = max(1, index.arity),
		call = lambda x, y = None: monadic_chain(links[(variadic_link(index, (x, y)) - 1) % (len(links) - 1)], x)
	),
	'ŀ': lambda index, links: attrdict(
		arity = 2,
		call = lambda x, y: dyadic_chain(links[(variadic_link(index, (x, y)) - 1) % (len(links) - 1)], (x, y))
	)
}

# Aliases

quicks['Ƈ'] = quicks['Ðf']
hypers['Ɱ'] = hypers['Ð€']
atoms ['Ẓ'] = atoms ['ÆP']

chain_separators = {
	'ø': (0, '', True),
	'µ': (1, '', True),
	')': (1, '€', True),
	'ð': (2, '', True),
	'ɓ': (2, '', False)
}
default_chain_separation = (-1, '', True)
str_arities = ''.join(chain_separators.keys())
str_strings = '“[^«»‘’”]*[«»‘’”]?'
str_charlit = '”.'
str_chrpair = '⁾..'
str_intpair = '⁽..'
str_realdec = '(?:0|-(?![1-9.])|-?\d*\.\d*|-?\d+)'
str_realnum = str_realdec.join(['(?:', '?ȷ', '?|', ')'])
str_complex = str_realnum.join(['(?:', '?ı', '?|', ')'])
str_literal = '(?:%s)' % '|'.join([str_strings, str_charlit, str_chrpair, str_complex, str_intpair])
str_litlist = '\[*' + str_literal + '(?:(?:\]*,\[*)' + str_literal + ')*' + '\]*'
str_nonlits = '|'.join(map(re.escape, list(atoms) + list(quicks) + list(hypers)))

regex_chain = re.compile('(?:^(?:' + str_nonlits + '|' + str_litlist + '| )+|[' + str_arities + '])(?:' + str_nonlits + '|' + str_litlist + '| )*')
regex_liter = re.compile(str_literal)
regex_token = re.compile(str_nonlits + '|' + str_litlist)
regex_flink = re.compile('(?=.)(?:[' + str_arities + ']|' + str_nonlits + '|' + str_litlist + '| )*¶?')
