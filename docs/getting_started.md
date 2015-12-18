# Getting started

The Jelly interpreter requires Python 3 and [SymPy].

To execute a Jelly program, you have three options:

1. `jelly f <file> [input]` reads the Jelly program stored in the specified file, using Jelly's own encoding.

	This option should be considered the default, but it exists solely for scoring puposes in code golf contests.
	
1. `jelly u <file> [input]` reads the Jelly program stored in the specified file, using the UTF-8 encoding.

1. `jelly e <code> [input]` reads the Jelly program as command line argument, using the UTF-8 encoding.

Alternatively, you can use the [Jelly interpreter] on [Try it online!]

Jelly's main input method is via command line arguments, although reading input from STDIN is also possible.

[Jelly interpreter]: http://jelly.tryitonline.net
[SymPy]: http://www.sympy.org/
[Try it online!]: http://tryitonline.net
