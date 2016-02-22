# Atoms

Atoms are Jelly's built-in links. **z** will denote the only argument of a monadic function, **x** and **y** the left and right arguments of a dyadic one.

Symbol|Arity|Function|Note
------|-----|--------|----
`¬`|1|Logical NOT.|Returns **1** or **0**.
`®`|0|Restore; retrieve the value of the register.|Initially 0.
`½`|1|Square root.
`×`|2|Multiplication.
`÷`|2|Floating point division.
`!`|1|Factorial or Pi function.
`%`|2|Modulus.
`&`|2|Bitwise AND.
`*`|2|Exponentiation with base **x**.
`+`|2|Addition.
`,`|2|Pair; return `[x, y]`.
`:`|2|Integer division.
`;`|2|Concatenate.
`<`|2|Less than.|Returns **1** or **0**.
`=`|2|Equals.|Returns **1** or **0**.
`>`|2|Greater than.|Returns **1** or **0**.
`A`|1|Absolute value.
`B`|1|Convert from integer to binary.
`C`|1|Complement; compute **1 - z**.
`D`|1|Convert from integer to decimal.
`F`|1|Flatten list.
`H`|1|Halve; compute **z ÷ 2**.
`I`|1|Increments; compute the deltas of consecutive elements of **z**.
`L`|1|Length.
`M`|1|Return all indices of **z** that correspond to maximal elements.|Indices are 1-based.
`N`|1|Negative; compute **-z**.
`O`|1|Ord; cast to number.
`P`|1|Product of a list.
`R`|1|Inclusive range.|Starts at `1`.<br>`-1R` gives `[-1, 0, 1]`.
`S`|1|Sum of a list.
`T`|1|Return all indices of **z** that correspond to truthy elements.|Indices are 1-based.
`U`|1|Upend; reverse an array.
`V`|1|Eval.
`W`|1|Wrap; return `[z]`.
`Z`|1|Zip; push the array of all columns of **z**.
`^`|2|Bitwise XOR.
`_`|2|Subtraction.
`a`|2|Logical AND.|Vectorizes with depth 0.
`b`|2|Convert from integer to base **y**.
`c`|2|Combinations; compute xCy.
`d`|2|Divmod.
`e`|2|Exists; return `x in y`.|Returns **1** or **0**.
`f`|2|Filter; remove the elements from **x** that are not in **y**.
`g`|2|Greatest common divisor.
`i`|2|Find the index of **y** in **x**.|Indices are 1-based.
`j`|2|Join list **x** with separator **y**.
`l`|2|Logarithm with base **y**.
`m`|2|Modular; return every **y** th element of **x**.
`o`|2|Logical OR.|Vectorizes with depth 0.
`p`|2|Cartesian product.
`r`|2|Inclusive range.|Descending if **x > y**.
`s`|2|Split **x** into slices of length **y**.
`t`|2|Trim all elements of **y** from both sides of **x**.
`x`|2|Times; repeat the elements of **x** **y** times.
`z`|2|Zip; transpose **x** with filler **y**.
`|`|2|Bitwise OR.
` ~ `|2|Bitwise NOT.
`°`|1|Convert **z** from degrees to radians.
`¹`|1|Identity; return **z**.
`²`|1|Square.
`³`|0|Return third command line argument (first input) or 256.
`⁴`|0|Return fourth command line argument (second input) or 16.
`⁵`|0|Return fifth command line argument (third input) or 10.
`⁶`|0|Return sixth command line argument (fourth input) or ' '.
`⁷`|0|Return seventh command line argument (fifth input) or '\n'.
`Ɠ`|0|Read and evaluate a single line from STDIN.
`ƈ`|0|Read a single character from STDIN.
`ɠ`|0|Read a single line from STDIN.
`Ḅ`|1|Convert from binary to integer.
`Ḍ`|1|Convert from decimal to integer.
`Ḥ`|1|Double; compute **2z**.
`Ṃ`|1|Minimum.
`Ọ`|1|Chr; cast to character.
`Ṣ`|1|Sort the list **z**.
`Ṭ`|1|Return a Boolean array with **1**'s at the indices in **z**.
`Ụ`|1|Grade the list **z** up, i.e., sort its indices by their values.
`Ḃ`|1|Bit; return `z % 2`.
`Ċ`|1|Ceil; round **z** up to the nearest integer.|Returns **z** for non-real **z**.
`Ḋ`|1|Dequeue; return `z[1:]`.
`Ė`|1|Enumerate.
`Ḟ`|1|Floor; round **z** down to the nearest integer.|Returns **z** for non-real **z**.
`Ḣ`|1|Head; pop and return the first element of **z**.|Modifies **z**.
`İ`|1|Inverse; compute **1 ÷ z**.
`Ṁ`|1|Maximum.
`Ṅ`|1|Print **z** and a linefeed.|Returns **z**.
`Ȯ`|1|Print **z**.|Returns **z**.
`Ṗ`|1|Pop; return `z[:-1]`.
`Q`|1|Return the unique elements of **z**, sorted by first appearance.
`Ṡ`|1|Sign of **z**.
`Ṫ`|1|Tail; pop and return the last element of **z**.|Modifies **z**.
`ạ`|2|Absolute difference.
`ḅ`|2|Convert from base **y** to integer.
`ḍ`|2|Check **y** for divisibility by **x**.|Return **1** or **0**.
`ị`|2|Return the element of **x** at index **y**.|Indices are 1-based.
`ḷ`|2|Left argument; return **x**.
`ṛ`|2|Right argument; return **y**.
`ṣ`|2|Split list **x** at occurrences of **y**.
`ȧ`|2|Logical AND.|Flat.
`ċ`|2|Count the occurrences of **y** in **x**.
`ḟ`|2|Filter; remove the elements from **x** that are in **y**.
`ḣ`|2|Head; return `x[:y]`.
`ȯ`|2|Logical OR.|Flat.
`ṗ`|2|Cartesian power.
`ṙ`|2|Rotate **x** **y** units to the left.
`ṡ`|2|Return all (overlapping) slices of length **y** of **x**.
`ṫ`|2|Tail; return `x[y - 1:]`.
`ẋ`|2|Repeat list **x** **y** times.
`ż`|2|Zip; interleave **x** and **y**.
`«`|2|Minimum of **x** and **y**.
`»`|2|Maximum of **x** and **y**.
`‘`|1|Increment; compute **z + 1**.
`’`|1|Decrement; compute **z - 1**.
`ØP`|0|Pi
`Øe`|0|Euler's number
`Æ½`|1|Compute the integer square root of **z**.
`ÆA`|1|Arccosine.
`ÆC`|1|Count the primes less or equal to **z**.
`ÆD`|1|Compute the array of **z**'s divisors.
`ÆE`|1|Compute the array of exponents of **z**'s prime factorization.|Includes zero exponents.
`ÆF`|1|Compute **z**'s prime factorization as **[prime, exponent]** pairs.
`ÆN`|1|Generate the **z**<sup>th</sup> prime.
`ÆP`|1|Test if **z** is a prime.|Returns **1** or **0**.
`ÆR`|1|Range; generate all primes between **2** and **z**.
`ÆS`|1|Sine.
`ÆT`|1|Tangent.
`Æe`|1|Exponential function.
`Æf`|1|Compute the array of primes whose product is **z**.
`Æl`|1|Natural logarithm.
`Æn`|1|Next; generate the closest prime strictly greater than **z**.
`Æp`|1|Previous; generate the closest prime strictly less than **z**.
`Ær`|1|Find the roots of a polynomial.|**z** is list of coefficients.
`ÆẠ`|1|Cosine.
`ÆẸ`|1|Inverse of `ÆE`.
`ÆṢ`|1|Arcsine.
`ÆṬ`|1|Arctangent.
`ÆṪ`|1|Totient function.
`Ær`|1|Construct the polynomial with roots **z**.|Returns list of coefficients.
`Æ°`|1|Convert **z** from radians to degrees.
`Æ²`|1|Test if **z** is a square.|Returns **1** or **0**.
`æ%`|2|Symmetric modulus; map **x** in the interval **(-y, y]**.
`æA`|2|Arctangent with two arguments, i.e., `atan2()`.
`Œ!`|1|All permutations of **z**.|May contain duplicates.
`ŒP`|1|Powerset of **z**.|May contain duplicates.
`Œp`|1|Cartesian product of **z**'s items.
`ŒḊ`|1|Depth.
`ŒṘ`|1|Python's string representation.
`œc`|2|Combinations without replacement.
`œl`|2|Trim all elements of **y** from the left side of **x**.
`œr`|2|Trim all elements of **y** from the right side of **x**.
`œ&`|2|Multiset intersection.
`œ-`|2|Multiset difference.
`œ^`|2|Multiset symmetric difference.
`œ|`|2|Multiset union.