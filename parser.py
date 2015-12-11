import re, jelly

str_arities = 'øµð'
str_strings = '“[^”]*”'
str_realdec = '(?:-?\d+(?:\.\d*)?|-?\d*\.\d+)'
str_realnum = str_realdec.join(['(?:', '?€', '?|', ')'])
str_complex = str_realnum.join(['(?:', '?ı', '?|', ')'])
str_literal = str_strings + '|' + str_complex
str_litlist = '\[*' + str_literal + '(?:(?:\]*,\[*)' + str_literal + ')*' + '\]*'
str_nonlits = '|'.join(map(re.escape, list(jelly.atoms.keys()) + list(jelly.hypers.keys()) + list(jelly.joints.keys())))

regex_chain = re.compile('(?:^|[' + str_arities + '])[^' + str_arities + ']+')
regex_token = re.compile(str_nonlits + '|' + str_litlist, flags = re.ASCII)

def parse_code(code):
	return [ [ parse_word(word) for word in regex_chain.findall(line) ] for line in code.split('\n') ]

def parse_word(word):
	chain = []
	arity = str_arities.find(word[0])
	for token in regex_token.findall(word):
		if token in jelly.atoms.keys():
			chain.append(jelly.atoms[token])
		elif token in jelly.hypers:
			chain.append(jelly.hypers[token](chain.pop()))
		elif token in jelly.joints:
			chain.append(jelly.joints[token]((chain.pop(), chain.pop)))
		else:
			chain.append(jelly.create_literal(token))
	return jelly.create_chain(chain, arity)