# Getting started

The Jelly interpreter requires Python 3 and [SymPy].

To execute a Jelly program, you have three options:

1. `jelly f <file> [input]` reads the Jelly program stored in the specified file, using the [Jelly code page].

	This option should be considered the default, but it exists solely for scoring puposes in code golf contests.
	
1. `jelly fu <file> [input]` reads the Jelly program stored in the specified file, using the UTF-8 encoding.

1. `jelly e <code> [input]` reads the Jelly program as command line argument, using the [Jelly code page].

	This requires setting the environment variable *LANG* (or your OS's equivalent) to *en_US* or compatible.

1. `jelly eu <code> [input]` reads the Jelly program as command line argument, using the UTF-8 encoding.

	This requires setting the environment variable *LANG* (or your OS's equivalent) to *en_US.UTF8* or compatible.

Alternatively, you can use the [Jelly interpreter] on [Try it online!]

Jelly's main input method is via command line arguments, although reading input from STDIN is also possible.

[Jelly code page]: https://github.com/DennisMitchell/jelly/blob/master/docs/code-page.md
[Jelly interpreter]: http://jelly.tryitonline.net
[SymPy]: http://www.sympy.org/
[Try it online!]: http://tryitonline.net
