# Atoms

Atoms are Jelly's built-in links. **z** will denote the only argument of a monadic function, **x** and **y** the left and right arguments of a dyadic one.

Symbol|Arity|Function|Note
------|-----|--------|----
`A`|1|Absolute value.
`a`|2|Logical AND.
`B`|1|Convert from integer to binary.
`Ḅ`|1|Convert from binary to integer.
`b`|2|Convert from integer to base **y**.
`ḅ`|2|Convert from base **y** to integer.
`C`|1|Complement; compute **1 - z**.
`c`|2|Combinations; compute xCy.
`D`|1|Convert from integer to decimal.
`Ḍ`|1|Convert from decimal to integer.
`g`|2|Greatest common divisor.
`H`|1|Halve; compute **z ÷ 2**.
`Ḥ`|1|Double; compute **2z**.
`I`|1|Inverse; compute **1 ÷ z**.
`L`|1|Length.
`l`|2|Logarithm with base **y**.
`N`|1|Negative; compute **-z**.
`O`|1|Ones; return the index of each **n** n times.
`o`|2|Logical OR.
`P`|1|Product of a list.
`R`|1|Inclusive range.|Starts at `1`. `-1R` gives `[-1, 0, 1]`.
`r`|2|Inclusive range.|Descending if **x > y**.
`S`|1|Sum of a list.
`Ṡ`|1|Sign of `z`.
`U`|1|Upend, reverse an array.
`x`|2|Times; repeat the elements of **x** **y** times.
`!`|1|Factorial or Pi function.
`<`|2|Less than.|Returns **1** or **0**.
`=`|2|Equals.|Returns **1** or **0**.
`>`|2|Greater than.|Returns **1** or **0**.
`:`|2|Integer division.
`;`|2|Concatenate.
`+`|2|Addition.
`_`|2|Subtraction.
`×`|2|Multiplication.
`÷`|2|Floating point division.
`%`|2|Modulus.
`*`|2|Exponentiation with base **x**.
`&`|2|Bitwise AND.
`^`|2|Bitwise XOR.
`|`|2|Bitwise OR.
`~`|2|Bitwise NOT.
`²`|1|Square.
`½`|1|Square root.
`¬`|1|Logical NOT.
`‘`|1|Increment; compute **z + 1**.
`’`|1|Decrement; compute **z - 1**.
`«`|2|Minimum of **x** and **y**.
`»`|2|Maximum of **x** and **y**.
`©`|1|Copy; save **z** in a special variable.
`®`|0|Restore; retrieve the value of the special variable.
`{`|2|Left argument; return **x**.
`¹`|1|Identity; retutrn **z**.
`}`|2|Right argument; return **y**.
`ÆC`|1|Count the primes less or equal to **z**.
`ÆD`|1|Compute the array of **z**'s divisors.
`ÆE`|1|Compute the array of exponents of **z**'s prime factorization.|Includes zero exponents.
`ÆẸ`|1|Inverse of `ÆE`.
`ÆF`|1|Compute **z**'s prime factorization as **[prime, exponent]** pairs.
`Æf`|1|Compute the array of primes whose product is **z**.
`ÆN`|1|Generate the **z**<sup>th</sup> prime.
`Æn`|1|Next; generate the closest prime strictly greater than **z**.
`ÆP`|1|Test if **z** is a prime.|Returns **1** or **0**.
`Æp`|1|Previous; generate the closest prime strictly lesser than **z**.
`ÆR`|1|Range; generate all primes between **2** and **z**.
`ÆT`|1|Totient function.
`Æ²`|1|Test if **z** is a square.|Returns **1** or **0**.
`Æ½`|1|Compute the integer square root of **z**.
`æ%`|2|Symmetric modulus; map **x** in the interval **(-y, y]**.
