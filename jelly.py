import fractions, numpy, operator

from helper import *

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
		ldepth = 0,
		call = abs
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
		call = lambda z: overload((math.ceil, identity), z)
	),
	'c': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: div(Pi(x), Pi(x - y) * Pi(y))
	),
	'ċ': attrdict(
		arity = 2,
		call = lambda x, y: iterable(x).count(y)
	),
	'ƈ': attrdict(
		arity = 0,
		call = lambda: sys.stdin.read(1)
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
		call = lambda z: z[1:]
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
		call = lambda x, y: int(y % x == 0)
	),
	'Ė': attrdict(
		arity = 1,
		call = lambda z: [[t + 1, u] for t, u in enumerate(iterable(z))]
	),
	'e': attrdict(
		arity = 2,
		call = lambda x, y: int(x in iterable(y))
	),
	'F': attrdict(
		arity = 1,
		call = flatten
	),
	'Ḟ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.floor, identity), z)
	),
	'f': attrdict(
		arity = 2,
		call = lambda x, y: [t for t in iterable(x) if t in iterable(y)]
	),
	'ḟ': attrdict(
		arity = 2,
		call = lambda x, y: [t for t in iterable(x) if not t in iterable(y)]
	),
	'g': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = fractions.gcd
	),
	'Ɠ': attrdict(
		arity = 0,
		call = lambda: eval(input())
	),
	'ɠ': attrdict(
		arity = 0,
		call = lambda: listify(input(), dirty = True)
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
		call = lambda z: z.pop(0) if z else 0
	),
	'ḣ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: x[:y]
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
	'i': attrdict(
		arity = 2,
		call = index
	),
	'ị': attrdict(
		arity = 2,
		ldepth = 0,
		call = lambda x, y: y[(int(x) - 1) % len(y)] if int(x) == x else [y[(math.floor(x) - 1) % len(y)], y[(math.ceil(x) - 1) % len(y)]]
	),
	'j': attrdict(
		arity = 2,
		call = lambda x, y: sum(([t, y] for t in x), [])[:-1]
	),
	'L': attrdict(
		arity = 1,
		call = len
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
		call = lambda z: [u + 1 for u, v in enumerate(z) if v == max(z)]
	),
	'Ṃ': attrdict(
		arity = 1,
		call = min
	),
	'Ṁ': attrdict(
		arity = 1,
		call = max
	),
	'm': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: iterable(x)[::y] if y else iterable(x) + iterable(x)[::-1]
	),
	'N': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: -z
	),
	'Ṅ': attrdict(
		arity = 1,
		call = lambda z: print(stringify(z)) or z
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
		call = lambda z: print(stringify(z), end = '') or z
	),
	'o': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = lambda x, y: x or y
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
		call = lambda z: z[:-1]
	),
	'p': attrdict(
		arity = 2,
		call = lambda x, y: listify(itertools.product(iterable(x, range = True), iterable(y, range = True)))
	),
	'ṗ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: listify(itertools.product(*([iterable(x, range = True)] * y)))
	),
	'Q': attrdict(
		arity = 1,
		call = unique
	),
	'R': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: list(range(1, int(z) + 1) or range(int(z), -int(z) + 1))
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
		call = lambda z: (z > 0) - (z < 0)
	),
	'Ṣ': attrdict(
		arity = 1,
		call = sorted
	),
	's': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: [x[i : i + y] for i in range(0, len(x), y)]
	),
	'ṡ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: [x[i : i + y] for i in range(len(x) - y + 1)]
	),
	'ṣ': attrdict(
		arity = 2,
		call = lambda x, y: listify(split_at(x, y))
	),
	'T': attrdict(
		arity = 1,
		call = lambda z: [u + 1 for u, v in enumerate(z) if v]
	),
	'Ṭ': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: [int(t + 1 in iterable(z)) for t in range(max(iterable(z)))]
	),
	'Ṫ': attrdict(
		arity = 1,
		call = lambda z: z.pop() if z else 0
	),
	't': attrdict(
		arity = 2,
		call = lambda x, y: trim(x, iterable(y), left = True, right = True)
	),
	'ṫ': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: x[y - 1 :]
	),
	'U': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: z[::-1]
	),
	'Ụ': attrdict(
		arity = 1,
		call = lambda z: sorted(range(1, len(z) + 1), key = lambda t: z[t - 1])
	),
	'W': attrdict(
		arity = 1,
		call = lambda z: [z]
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
	'Z': attrdict(
		arity = 1,
		call = lambda z: listify(map(lambda t: filter(None.__ne__, t), itertools.zip_longest(*map(iterable, z))))
	),
	'z': attrdict(
		arity = 2,
		call = lambda x, y: listify(itertools.zip_longest(*map(iterable, x), fillvalue = y))
	),
	'ż': attrdict(
		arity = 2,
		call = lambda x, y: listify(map(lambda z: filter(None.__ne__, z), itertools.zip_longest(iterable(x), iterable(y))))
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
		call = lambda z: overload((math.cos, cmath.cos), z)
	),
	'ÆẠ': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.acos, cmath.acos), z)
	),
	'ÆC': attrdict(
		arity = 1,
		ldepth = 0,
		call = sympy.ntheory.generate.primepi
	),
	'ÆD': attrdict(
		arity = 1,
		ldepth = 0,
		call = sympy.ntheory.factor_.divisors
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
		call = lambda z: [[x, y] for x, y in sympy.ntheory.factor_.factorint(z).items()]
	),
	'Æe': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.exp, cmath.exp), z)
	),
	'Æf': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: rld(sympy.ntheory.factor_.factorint(z).items())
	),
	'Æl': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: overload((math.log, cmath.log), z)
	),
	'ÆN': attrdict(
		arity = 1,
		ldepth = 0,
		call = sympy.ntheory.generate.prime
	),
	'Æn': attrdict(
		arity = 1,
		ldepth = 0,
		call = sympy.ntheory.generate.nextprime
	),
	'ÆP': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: int(sympy.primetest.isprime(z))
	),
	'Æp': attrdict(
		arity = 1,
		ldepth = 0,
		call = sympy.ntheory.generate.prevprime
	),
	'ÆR': attrdict(
		arity = 1,
		ldepth = 0,
		call = lambda z: list(sympy.ntheory.generate.primerange(2, z + 1))
	),
	'Ær': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: list(numpy.roots(z[::-1]))
	),
	'Æṛ': attrdict(
		arity = 1,
		ldepth = 1,
		call = lambda z: list(numpy.poly(z))[::-1]
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
	'æA': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = math.atan2
	),
	'Œ!': attrdict(
		arity = 1,
		call = lambda z: listify(itertools.permutations(iterable(z, range = True)))
	),
	'ŒḊ': attrdict(
		arity = 1,
		call = depth
	),
	'ŒP': attrdict(
		arity = 1,
		call = powerset
	),
	'Œp': attrdict(
		arity = 1,
		call = lambda z: listify(itertools.product(*[iterable(t, range = True) for t in z]))
	),
	'ŒṘ': attrdict(
		arity = 1,
		call = lambda z: listify(repr(z))
	),
	'æ%': attrdict(
		arity = 2,
		ldepth = 0,
		rdepth = 0,
		call = symmetric_mod
	),
	'œc': attrdict(
		arity = 2,
		rdepth = 0,
		call = lambda x, y: listify(itertools.combinations(iterable(x, range = True), y))
	),
	'œl': attrdict(
		arity = 2,
		call = lambda x, y: trim(x, iterable(y), left = True)
	),
	'œr': attrdict(
		arity = 2,
		call = lambda x, y: trim(x, iterable(y), right = True)
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
	'ØP': attrdict(
		arity = 0,
		call = lambda: math.pi
	),
	'Øe': attrdict(
		arity = 0,
		call = lambda: math.e
	)
}

quicks = {
	'©': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x = None, y = None: copy(atoms['®'], variadic_link(links[0], (x, y)))
		)]
	),
	'ß': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = -1,
			call = lambda x = None, y = None: variadic_chain(outmost_links[index], (x, y))
		)]
	),
	'¢': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 0,
			call = lambda: niladic_chain(outmost_links[index - 1])
		)]
	),
	'Ç': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 1,
			call = lambda z: monadic_chain(outmost_links[index - 1], z)
		)]
	),
	'ç': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 2,
			call = lambda x, y: dyadic_chain(outmost_links[index - 1], (x, y))
		)]
	),
	'Ñ': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 1,
			call = lambda z: monadic_chain(outmost_links[(index + 1) % len(outmost_links)], z)
		)]
	),
	'ñ': attrdict(
		condition = lambda links: True,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 2,
			call = lambda x, y: dyadic_chain(outmost_links[(index + 1) % len(outmost_links)], (x, y))
		)]
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
	'¤': attrdict(
		condition = lambda links: len(links) > 1 and links[0].arity == 0,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 0,
			call = lambda: niladic_chain(links)
		)]
	),
	'$': attrdict(
		condition = lambda links: len(links) > 1 and not leading_constant(links),
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 1,
			call = lambda z: monadic_chain(links, z)
		)]
	),
	'¥': attrdict(
		condition = lambda links: len(links) > 1 and not leading_constant(links),
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = 2,
			call = lambda x, y: dyadic_chain(links, (x, y))
		)]
	),
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
	'Ðf': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x, y = None: list(filter(lambda t: variadic_link(links[0], (t, y)), x))
		)]
	),
	'Ðḟ': attrdict(
		condition = lambda links: links,
		quicklink = lambda links, outmost_links, index: [attrdict(
			arity = links[0].arity,
			call = lambda x, y = None: list(itertools.filterfalse(lambda t: variadic_link(links[0], (t, y)), x))
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
	)
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
	'/': lambda link, none = None: attrdict(
		arity = 1,
		call = lambda z: functools.reduce(lambda x, y: dyadic_link(link, (x, y)), iterable(z))
	),
	'\\': lambda link, none = None: attrdict(
		arity = 1,
		call = lambda z: list(itertools.accumulate(iterable(z), lambda x, y: dyadic_link(link, (x, y))))
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
		arity = link.arity,
		call = lambda x, y = None: [variadic_link(link, (t, y)) for t in iterable(x)]
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