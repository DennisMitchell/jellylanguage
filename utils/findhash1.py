from hashlib import shake_256
from jelly import jellify, jelly_eval
from sys import stdin, stdout

objects = stdin.read()

try:
	objects = jellify(eval(objects))
except:
	objects = jelly_eval(objects, [])

for object in objects:
	stdout.buffer.write(shake_256(repr(object).encode('utf-8')).digest(512))
