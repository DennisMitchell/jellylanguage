# Jelly

Jelly is a golfing language inspired by J.

### Documentation

* [Tutorial]
* [Code page]
* [Atoms]
* [Quicks]
* [Syntax]

### Quickstart

The Jelly interpreter requires Python 3.

To download, install, and use Jelly, proceed as follows.

```
$ git clone -q https://github.com/DennisMitchell/jellylanguage.git
$ cd jellylanguage
$ pip3 install --upgrade --user .
$ jelly eun '“3ḅaė;œ»'
Hello, World!
$ jelly eun '×' 14 3
42
$ jelly
Usage:

    jelly f <file> [input]    Reads the Jelly program stored in the
                              specified file, using the Jelly code page.
                              This option should be considered the default,
                              but it exists solely for scoring purposes in
                              code golf contests.

    jelly fu <file> [input]   Reads the Jelly program stored in the
                              specified file, using the UTF-8 encoding.

    jelly e <code> [input]    Reads a Jelly program as a command line
                              argument, using the Jelly code page. This
                              requires setting the environment variable
                              LANG (or your OS's equivalent) to en_US or
                              compatible.

    jelly eu <code> [input]   Reads a Jelly program as a command line
                              argument, using the UTF-8 encoding. This
                              requires setting the environment variable
                              LANG (or your OS's equivalent) to en_US.UTF8
                              or compatible.

    Append an `n` to the flag list to append a trailing newline to the
    program's output.

Visit http://github.com/DennisMitchell/jellylanguage for more information.
```

Alternatively, you can use the [Jelly interpreter] on [Try It Online].

Jelly's main input method is via command line arguments, although reading input from STDIN is also possible.

[Atoms]: https://github.com/DennisMitchell/jellylanguage/wiki/Atoms
[Code page]: https://github.com/DennisMitchell/jellylanguage/wiki/Code-page
[Jelly interpreter]: https://tio.run/#jelly
[Quicks]: https://github.com/DennisMitchell/jellylanguage/wiki/Quicks
[Syntax]: https://github.com/DennisMitchell/jellylanguage/wiki/Syntax
[Try It Online]: https://tryitonline.net
[Tutorial]: https://github.com/DennisMitchell/jellylanguage/wiki/Tutorial
