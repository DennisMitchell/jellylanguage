import re, jelly

str_arities = 'øµð'
str_strings = '“[^”]*”?'
str_realdec = '(?:0|-?\d+(?:\.\d*)?|-?\d*\.\d+|-)'
str_realnum = str_realdec.join(['(?:', '?ȷ', '?|', ')'])
str_complex = str_realnum.join(['(?:', '?ı', '?|', ')'])
str_literal = '(?:' + str_strings + '|' + str_complex + ')'
str_litlist = '\[*' + str_literal + '(?:(?:\]*,\[*)' + str_literal + ')*' + '\]*'
str_nonlits = '|'.join(map(re.escape, list(jelly.atoms) + list(jelly.actors) + list(jelly.hypers) + list(jelly.joints) + list(jelly.nexus)))

regex_chain = re.compile('(?:^|[' + str_arities + '])[^' + str_arities + ']+')
regex_liter = re.compile(str_literal)
regex_token = re.compile(str_nonlits + '|' + str_litlist)
regex_flink = re.compile('[^¶]*¶|[^¶]+¶?')

def parse_code(code):
	lines = regex_flink.findall(code)
	links = [[] for line in lines]
	for index, line in enumerate(lines):
		chains = links[index]
		for word in regex_chain.findall(line):
			chain = []
			arity = str_arities.find(word[0])
			for token in regex_token.findall(word):
				if token in jelly.atoms.keys():
					chain.append(jelly.atoms[token])
				elif token in jelly.actors:
					chain.append(jelly.actors[token](index, links))
				elif token in jelly.hypers:
					x = chain.pop() if chain else chains.pop()
					chain.append(jelly.hypers[token](x, links))
				elif token in jelly.joints:
					y = chain.pop() if chain else chains.pop()
					x = chain.pop() if chain else chains.pop()
					chain.append(jelly.joints[token]((x, y)))
				elif token in jelly.nexus:
					z = chain.pop() if chain else chains.pop()
					y = chain.pop() if chain else chains.pop()
					x = chain.pop() if chain else chains.pop()
					chain.append(jelly.nexus[token]((x, y, z)))
				else:
					chain.append(jelly.create_literal(regex_liter.sub(parse_literal, token)))
			chains.append(jelly.create_chain(chain, arity))
	return links

def parse_literal(literal_match):
	literal = literal_match.group(0).replace('¶', '\n')
	if '“' in literal:
		parsed = literal.rstrip('”').split('“')[1:]
		if len(parsed) == 1:
			parsed = parsed[0]
	else:
		parsed = eval('+ 1j *'.join([
			repr(eval('* 10 **'.join(['-1' if part == '-' else part or repr(2 * index + 1) for index, part in enumerate(component.split('ȷ'))])) if component else index)
			for index, component in enumerate(literal.split('ı'))
		]))
	return repr(parsed) + ' '