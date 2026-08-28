The core intuition is:

\[
\boxed{\text{The polynomial hides a seven-qubit “thickness fiber” under every site of }C_8\times C_4.}
\]

The generator was designed to make the uniform vector on each fiber a logical operator, while coupling the fibers strongly enough that nothing lighter survives.

## 1. Physical lattice versus logical lattice

The physical group is

\[
G=C_{56}\times C_4.
\]

Inside \(C_{56}\), define

\[
d=x^8.
\]

Since \(d\) has order seven,

\[
D=\langle d\rangle\cong C_7.
\]

The quotient is

\[
G/D\cong C_8\times C_4,
\]

which is exactly the desired 32-site checkerboard sublattice.

The orbit sum

\[
\omega_D=1+d+d^2+\cdots+d^6
\]

is the uniform support on one seven-qubit thickness fiber. Its translates

\[
z_{a,b}=x^a y^b\omega_D,
\qquad (a,b)\in C_8\times C_4,
\]

are the 32 logical supports on each BB half.

Because cosets of a subgroup are disjoint, these supports automatically partition the physical qubits. That is where the parallel-STAR property comes from.

## 2. Why every generator term is paired

Write the selected polynomial using \(d=x^8\):

\[
a=(1+d)+x(1+d^3)+dy(1+d^2).
\]

Every piece has the form

\[
q(1+d^r),
\]

where \(q\) positions the pair on the quotient lattice.

The key identity is

\[
d^r\omega_D=\omega_D,
\]

so

\[
(1+d^r)\omega_D
=\omega_D+\omega_D
=0
\]

over \(\mathbb F_2\). Therefore,

\[
a\,z_{a,b}=0.
\]

The adjoint \(a^\dagger\) has the same property because negative thickness translations also leave \(\omega_D\) invariant.

This is analogous to a discrete derivative: \(1+d^r\) kills anything constant along the thickness direction. The logical fibers are precisely constant along that direction.

Because \(m=7\) is odd,

\[
\langle z_{a,b},z_{a,b}\rangle
=7\bmod 2
=1.
\]

Thus \(X(z_{a,b})\) and \(Z(z_{a,b})\) anticommute. That immediately proves each fiber support is a nontrivial logical qubit rather than a stabilizer.

## 3. Why there are three pairs

The three paired terms occupy the following quotient locations:

| Paired term | Expanded pair | Quotient location | Thickness step |
|-------------|---------------|-------------------|---------------:|
| 1+d         | 1+x^8         | (0,0)             |              1 |
| x(1+d^3)    | x+x^25        | (1,0)             |              3 |
| dy(1+d^2)   | x^8y+x^24y    | (0,1)             |              2 |

After quotienting out \(D\), the three locations are

\[
(0,0),\quad(1,0),\quad(0,1).
\]

Their differences generate both axes of \(C_8\times C_4\).

That is crucial. With only two paired terms, there is only one quotient displacement. A single element generates a cyclic subgroup, but \(C_8\times C_4\) is not cyclic. Consequently, every weight-eight candidate split into at least four disconnected components.

The third pair supplies the second independent direction:

\[
(1,0)\quad\text{and}\quad(0,1).
\]

That connects the complete logical quotient. This is why the minimum connected ansatz had six monomials in \(a\), producing weight-12 stabilizer checks.

## 4. Why the particular thickness steps work

The steps \(1,3,2\) are all nonzero modulo seven. Because seven is prime, each one individually traverses the entire thickness cycle.

Using several different steps also makes short cancellations harder. Informally, the checks compare different offsets within neighboring fibers rather than repeatedly making the same comparison.

This part was not guaranteed solely by the ansatz:

- The paired form guaranteed the desired fibers.
- The three quotient locations guaranteed two-dimensional connectivity.
- The search checked that there were no accidental logical sectors.
- The exact optimization proved that no logical operator of weight below seven exists.

So the specific \(1,3,2\) choice is not uniquely magical; it is one of many members of the structured family that avoids low-weight degeneracies. In fact, most accepted weight-12 candidates had distance seven.

## 5. Why it is ZX dual

Let \(A\) be the binary convolution matrix for \(a\). We choose the second BB generator to be

\[
b=a^\dagger,
\]

whose matrix is \(A^T\). The GALA/BB checks then become

\[
H_X=[A\mid A^T],
\qquad
H_Z=[A\mid A^T].
\]

Therefore,

\[
H_X=H_Z.
\]

Moreover, because the group is abelian,

\[
AA^T=A^TA,
\]

and hence

\[
H_XH_Z^T
=AA^T+A^TA
=0
\]

over \(\mathbb F_2\).

This gives exact physical Hadamard symmetry:

\[
H^{\otimes448}:X\leftrightarrow Z.
\]

Since the logical \(X\) and \(Z\) representatives use the same fiber support, there is no hidden logical permutation:

\[
\bar X_{a,b}^{L/R}\longleftrightarrow
\bar Z_{a,b}^{L/R}.
\]

The weight-12 checks are also doubly even, so physical transversal \(S\) preserves the stabilizers. Because the logical fibers have weight \(7\equiv3\pmod4\), its logical action is uniformly \(S^\dagger\).

## 6. Where the translation automorphisms come from

Group-ring convolution is translation invariant. Multiplication by \(x\) or \(y\) simply relabels the check rows and physical qubits, so simultaneous translations of both BB halves preserve the stabilizer code.

On the logical fibers,

\[
x:z_{a,b}\mapsto z_{a+1,b},
\qquad
y:z_{a,b}\mapsto z_{a,b+1}.
\]

Although physical \(x\) has order 56,

\[
x^8z_{a,b}=z_{a,b},
\]

because \(x^8=d\) only moves within the thickness fiber. Thus \(x\) has logical order eight. Similarly, \(y\) has order four.

This gives the exact logical action

\[
C_8\times C_4.
\]

The action is diagonal on the two sectors:

\[
T_x=T_x^{(L)}\otimes T_x^{(R)},
\qquad
T_y=T_y^{(L)}\otimes T_y^{(R)}.
\]

The stabilizers tie the two physical halves together, which is why translating only \(L\) or only \(R\) fails. The halves are independent logical sectors, but not independently movable physical codes.

There is also a half-swap-plus-inversion symmetry. Inversion sends \(A\leftrightarrow A^T\), so swapping the two halves at the same time preserves \([A\mid A^T]\).

## 7. Why the distance lands exactly at seven

Every fiber logical has weight seven, so immediately

\[
d\leq7.
\]

The check constraints couple every quotient position in both directions and mix the seven thickness coordinates using steps \(1,2,3\). For this candidate, that eliminates all nontrivial logical errors of weights one through six.

The exact solver certified a minimum of seven in one \(L\)-sector class and one \(R\)-sector class. Translation symmetry relates each result to all 32 classes in the corresponding half. Because the 64 fibers form a complete logical basis, those two calculations cover every possible nontrivial logical class.

Thus,

\[
d=7.
\]

So the distance equals the thickness because the construction deliberately creates weight-seven logical fibers, and the polynomial is sufficiently connected to prevent anything lighter.

## 8. The division between design and search

The algebraic design guaranteed:

- 32 disjoint fibers on each half;
- logical \(C_8\times C_4\) translations;
- ZX duality;
- transversal Hadamard;
- weight-12 checks and even syndrome parity;
- disjoint STAR supports.

The search and certification established:

- the Tanner graph is connected;
- the fibers account for exactly all 64 logical qubits;
- there are no accidental logical sectors;
- no logical operator has weight below seven.

In one sentence: the factors \(1+d^r\) create the logical fibers, the three quotient locations connect the two-dimensional Ising lattice, and the adjoint pairing creates the ZX-dual BB stabilizer structure.

Usage updated: tokens +82.5k



> What is a "fiber" and "thickness"?  How are those defined?

------------------------------------------------------------------------

“Fiber” and “thickness” are descriptive terms we introduced for this construction; they are not special qLDPC data types.

The precise object is the quotient map

\[
\pi:G\longrightarrow G/D,
\]

where

\[
G=C_{56}\times C_4,
\qquad
D=\langle x^8\rangle\cong C_7.
\]

## Fiber

A fiber is the set of physical group coordinates that map to the same logical quotient coordinate.

Writing a physical coordinate as

\[
(j,b)\in \mathbb Z_{56}\times\mathbb Z_4,
\]

the quotient map is

\[
\pi(j,b)=(j\bmod 8,b).
\]

Therefore, the fiber above the logical site \((a,b)\in C_8\times C_4\) is

\[
\begin{aligned}
\pi^{-1}(a,b)
&=(a,b)+D\\
&=\left\{
(a+8r\bmod56,b):r=0,\ldots,6
\right\}.
\end{aligned}
\]

For example,

\[
\pi^{-1}(2,1)=
\{(2,1),(10,1),(18,1),(26,1),(34,1),(42,1),(50,1)\}.
\]

These seven physical coordinates all have the same logical label \((2,1)\).

In group-ring notation, the indicator vector of the fiber above \((a,b)\) is

\[
z_{a,b}
=
x^a y^b
\underbrace{
\left(1+x^8+x^{16}+\cdots+x^{48}\right)
}_{\omega_D}.
\]

This seven-qubit support becomes the canonical support of both \(\bar X_{a,b}\) and \(\bar Z_{a,b}\).

## Thickness

“Thickness” refers to the extra subgroup direction \(D\) that is removed when we pass to the logical quotient.

Here,

\[
D=\{1,x^8,x^{16},\ldots,x^{48}\},
\qquad |D|=7.
\]

Thus the thickness is seven: every logical quotient site has seven physical coordinates lying above it.

Equivalently, because \(8\) and \(7\) are coprime,

\[
C_{56}\cong C_8\times C_7,
\]

so conceptually

\[
G
=C_{56}\times C_4
\cong
\underbrace{C_8\times C_4}_{\text{logical lattice}}
\times
\underbrace{C_7}_{\text{thickness}}.
\]

We use \(C_{56}\times C_4\) in the code construction so the physical lift still appears as a product of only two cyclic groups.

## Physical counting

On each BB half:

\[
32\text{ quotient sites}\times7\text{ physical positions per fiber}
=224\text{ physical qubits}.
\]

The two BB halves give

\[
224_L+224_R=448
\]

physical qubits.

Each half consequently contains 32 disjoint fiber supports:

\[
\begin{array}{c}
\text{224 physical qubits on one half}\\
\downarrow\text{ partition into cosets of }D\\
32\text{ fibers of size }7\\
\downarrow\\
32\text{ canonical logical qubits}.
\end{array}
\]

## Why the fibers become logical operators

The polynomial generator is built from terms containing

\[
1+x^{8r}.
\]

Translation by \(x^{8r}\) only moves around inside a fiber. The uniform fiber vector is invariant:

\[
x^{8r}\omega_D=\omega_D.
\]

Therefore,

\[
(1+x^{8r})\omega_D
=\omega_D+\omega_D
=0.
\]

This places every translated fiber vector \(z_{a,b}\) in the kernel of the appropriate check matrix.

Because the fiber contains seven qubits,

\[
z_{a,b}\cdot z_{a,b}=7\bmod2=1.
\]

Hence the same-support operators \(X(z_{a,b})\) and \(Z(z_{a,b})\) anticommute and define a logical qubit.

One caveat: the code is not simply 32 separate seven-qubit repetition codes. The fibers describe the logical supports and quotient structure, but the weight-12 stabilizers couple different fibers together. That coupling is what prevents smaller logical operators and produces distance seven.

Usage updated: tokens +84.5k



> How can I see that the 32 qubits are arranged in a grid, such that the translation generators guenuinly map them consistently?

------------------------------------------------------------------------

The important distinction is that the \(8\times4\) grid is a coordinate grid for one checkerboard color. It is not a rectangular subset of the original \(8\times8\) lattice—nearest neighbors always lie in the opposite checkerboard color.

## 1. The code gives an abstract \(8\times4\) torus

The logical supports are indexed by cosets:

\[
F_{a,b}
=
\left\{
(a+8r,b):r\in\mathbb Z_7
\right\},
\qquad
(a,b)\in\mathbb Z_8\times\mathbb Z_4.
\]

Physical multiplication by \(x\) sends

\[
xF_{a,b}=F_{a+1,b},
\]

and physical multiplication by \(y\) sends

\[
yF_{a,b}=F_{a,b+1}.
\]

This remains true at the boundaries. For example,

\[
\begin{aligned}
xF_{7,b}
&=\{8+8r:r\in\mathbb Z_7\}\\
&=\{8r:r\in\mathbb Z_7\}\\
&=F_{0,b}.
\end{aligned}
\]

Thus, on the logical supports,

\[
T_x^8=I,\qquad T_y^4=I,\qquad T_xT_y=T_yT_x.
\]

This is precisely the regular translation action of

\[
C_8\times C_4.
\]

So the 32 logical qubits form the Cayley grid of \(C_8\times C_4\): \(T_x\) moves one column and \(T_y\) moves one row, both periodically.

## 2. Embed that grid into one checkerboard color

Label the ordinary periodic square lattice by

\[
(u,v)\in\mathbb Z_8\times\mathbb Z_8.
\]

Define the embedding of the logical grid into the even checkerboard sites by

\[
\boxed{
\phi_A(a,b)=(a,\ a+2b)\pmod8.
}
\]

Every point is even because

\[
u+v=a+(a+2b)=2(a+b)\equiv0\pmod2.
\]

It is also one-to-one. If

\[
(a,a+2b)=(a',a'+2b')\pmod8,
\]

then \(a=a'\pmod8\), followed by \(b=b'\pmod4\). Since there are 32 labels and 32 even sites, this is a bijection.

The complete correspondence is:

| b \ a | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     |
|------:|-------|-------|-------|-------|-------|-------|-------|-------|
|     0 | (0,0) | (1,1) | (2,2) | (3,3) | (4,4) | (5,5) | (6,6) | (7,7) |
|     1 | (0,2) | (1,3) | (2,4) | (3,5) | (4,6) | (5,7) | (6,0) | (7,1) |
|     2 | (0,4) | (1,5) | (2,6) | (3,7) | (4,0) | (5,1) | (6,2) | (7,3) |
|     3 | (0,6) | (1,7) | (2,0) | (3,1) | (4,2) | (5,3) | (6,4) | (7,5) |

These are exactly the 32 even-parity sites.

## 3. The translations are consistent with this embedding

Under the embedding,

\[
\begin{aligned}
\phi_A(a+1,b)
&=(a+1,a+1+2b)\\
&=\phi_A(a,b)+(1,1),
\end{aligned}
\]

so logical \(T_x\) is the square-lattice diagonal translation

\[
(u,v)\mapsto(u+1,v+1).
\]

Similarly,

\[
\begin{aligned}
\phi_A(a,b+1)
&=(a,a+2b+2)\\
&=\phi_A(a,b)+(0,2),
\end{aligned}
\]

so logical \(T_y\) is

\[
(u,v)\mapsto(u,v+2).
\]

Therefore the following diagram is consistent:

\[
\begin{array}{ccc}
F_{a,b} & \xrightarrow{\text{physical }x} & F_{a+1,b}\\
\downarrow && \downarrow\\
(a,b) & \xrightarrow{T_x} & (a+1,b)\\
\downarrow\phi_A && \downarrow\phi_A\\
(u,v) & \xrightarrow{+(1,1)} & (u+1,v+1).
\end{array}
\]

There is an analogous diagram for \(y\) and the square translation \((0,2)\).

This shows that the grid structure is not an arbitrary relabeling: the physical code permutations, quotient translations, and square-lattice translations all intertwine.

## 4. Add the odd checkerboard block

Identify the odd site with the east neighbor of the corresponding even site:

\[
\phi_B(a,b)=\phi_A(a,b)+(1,0)
=(a+1,a+2b).
\]

Then the east bond from \(A(a,b)\) goes directly to \(B(a,b)\).

The other neighbors are obtained by fixed logical translations of the entire \(B\) block:

| Bond from A(a,b) | Destination in B coordinates | Shift of block B |
|------------------|------------------------------|------------------|
| East             | B(a,b)                       | I                |
| West             | B(a-2,b+1)                   | T_x^-2 T_y       |
| North            | B(a-1,b+1)                   | T_x^-1 T_y       |
| South            | B(a-1,b)                     | T_x^-1           |

For example,

\[
\begin{aligned}
\phi_B(a-2,b+1)
&=(a-1,a-2+2b+2)\\
&=(a-1,a+2b)\\
&=\phi_A(a,b)+(-1,0),
\end{aligned}
\]

so \(T_x^{-2}T_y\) genuinely maps every even site to its west neighbor—not just one selected site. The same group element works uniformly for all 32 sites because these are translation-equivariant coordinate maps.

Thus the construction has three compatible levels:

\[
\boxed{
\begin{aligned}
&\text{physical code translation}\\
&\quad\downarrow\\
&\text{logical }C_8\times C_4\text{ translation}\\
&\quad\downarrow\\
&\text{uniform checkerboard bond matching}.
\end{aligned}
}
\]

The executable checks for these relationships are included in Sections 5 and 6 of ising_conditions.ipynb.
