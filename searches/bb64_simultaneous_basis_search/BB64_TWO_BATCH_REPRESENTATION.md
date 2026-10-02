# Preferred two-batch representation of the `[[64,8,8]]` BB code

This document fixes one complete physical presentation of the Liang--Chen
self-dual bivariate-bicycle code used for the `C_4 x C_2` Ising block.  All
indices are zero based.  It was generated from the preferred candidate-0 basis
and the canonical simultaneous syndrome schedule.

## Source identity

- Basis SHA-256: `8cbf96d32e5e16b357c2a4b751a94f09b3e136d8d0d722226a267be3395ad03e`
- Schedule SHA-256: `0d07ff49c000900b65928fd5f26cb463315448ffbf2b42c4b822ba3d68fc656c`
- Schedule ID: `67e327dc7cb35335`
- Code parameters: `[[64,8,8]]`
- Check presentation: $H_X=H_Z$, with 32 displayed rows of each type, rank 28,
  and row weight 8.

## Physical data-qubit labels

The data qubits are $q_0,q_1,\ldots,q_{63}$.  The first BB half is
$L=\{q_0,\ldots,q_{31}\}$ and the second is
$R=\{q_{32},\ldots,q_{63}\}$.  In the stored twisted-torus coordinates,

\[
q(L,x,y)=q_{8x+y},\qquad q(R,x,y)=q_{32+8x+y},
\qquad x\in\mathbb Z_4,\ y\in\mathbb Z_8.
\]

| Half and first coordinate | `y=0` | `y=1` | `y=2` | `y=3` | `y=4` | `y=5` | `y=6` | `y=7` |
|---------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| `L, x=0`                  | `q0`  | `q1`  | `q2`  | `q3`  | `q4`  | `q5`  | `q6`  | `q7`  |
| `L, x=1`                  | `q8`  | `q9`  | `q10` | `q11` | `q12` | `q13` | `q14` | `q15` |
| `L, x=2`                  | `q16` | `q17` | `q18` | `q19` | `q20` | `q21` | `q22` | `q23` |
| `L, x=3`                  | `q24` | `q25` | `q26` | `q27` | `q28` | `q29` | `q30` | `q31` |
| `R, x=0`                  | `q32` | `q33` | `q34` | `q35` | `q36` | `q37` | `q38` | `q39` |
| `R, x=1`                  | `q40` | `q41` | `q42` | `q43` | `q44` | `q45` | `q46` | `q47` |
| `R, x=2`                  | `q48` | `q49` | `q50` | `q51` | `q52` | `q53` | `q54` | `q55` |
| `R, x=3`                  | `q56` | `q57` | `q58` | `q59` | `q60` | `q61` | `q62` | `q63` |

These coordinates label the code artifact.  They do not assert that the
neutral-atom array must be embedded with this twisted periodic boundary in the
laboratory plane.

## Logical X and Z representatives

For a binary support $S\subseteq\{0,\ldots,63\}$, define

\[
Z(S)=\prod_{j\in S} Z_{q_j},\qquad
X(S)=\prod_{j\in S} X_{q_j}.
\]

The saved physical representatives are:

| Logical | Z support                                 | X support                                 | After transversal H |
|---------|-------------------------------------------|-------------------------------------------|---------------------|
| `0`     | `{q6, q22, q24, q25, q31, q38, q48, q54}` | `{q2, q3, q9, q10, q16, q35, q48, q63}`   | `Z0 -> X5`          |
| `1`     | `{q0, q1, q8, q15, q22, q33, q54, q61}`   | `{q0, q18, q20, q32, q41, q42, q43, q52}` | `Z1 -> X4`          |
| `2`     | `{q7, q18, q26, q39, q40, q41, q49, q50}` | `{q3, q4, q10, q11, q17, q36, q49, q56}`  | `Z2 -> X7`          |
| `3`     | `{q7, q19, q21, q39, q44, q45, q46, q51}` | `{q5, q21, q24, q30, q31, q37, q53, q55}` | `Z3 -> X6`          |
| `4`     | `{q0, q18, q20, q32, q41, q42, q43, q52}` | `{q0, q1, q8, q15, q22, q33, q54, q61}`   | `Z4 -> X1`          |
| `5`     | `{q2, q3, q9, q10, q16, q35, q48, q63}`   | `{q6, q22, q24, q25, q31, q38, q48, q54}` | `Z5 -> X0`          |
| `6`     | `{q5, q21, q24, q30, q31, q37, q53, q55}` | `{q7, q19, q21, q39, q44, q45, q46, q51}` | `Z6 -> X3`          |
| `7`     | `{q3, q4, q10, q11, q17, q36, q49, q56}`  | `{q7, q18, q26, q39, q40, q41, q49, q50}` | `Z7 -> X2`          |

Every displayed logical representative has physical weight eight.  The eight
pairs are canonical:

\[
\bar Z_i\bar X_j=(-1)^{\delta_{ij}}\bar X_j\bar Z_i.
\]

## Disjoint logical-Z batches

The eight Z logicals are covered exactly once by two batches:

\[
\mathcal B_Z^{(0)}=\{0,3,4,7\},\qquad
\mathcal B_Z^{(1)}=\{1,2,5,6\}.
\]

Their union supports are

\[
\begin{aligned}
\operatorname{supp}\!\left(\mathcal B_Z^{(0)}\right)
  &=\{0,3,4,6,7,10,11,17,18,19,20,21,22,24,25,31,32,36,38,39,41,42,43,44,45,46,48,49,51,52,54,56\},\\
\operatorname{supp}\!\left(\mathcal B_Z^{(1)}\right)
  &=\{0,1,2,3,5,7,8,9,10,15,16,18,21,22,24,26,30,31,33,35,37,39,40,41,48,49,50,53,54,55,61,63\}.
\end{aligned}
\]

Each union contains 32 qubits because the four weight-eight logical supports
inside that batch are pairwise disjoint.  Supports in different batches are
allowed to overlap, which is why the batches are executed separately.

For completeness, the corresponding internally disjoint X batches are

\[
\mathcal B_X^{(0)}=\{1,2,5,6\},\qquad
\mathcal B_X^{(1)}=\{0,3,4,7\}.
\]

## Transversal Hadamard and logical permutations

The exact support identity is

\[
\operatorname{supp}(\bar X_i)=\operatorname{supp}(\bar Z_{p(i)}),
\qquad
p=(0\;5)(1\;4)(2\;7)(3\;6).
\]

Consequently,

\[
H^{\otimes64}:\quad
\bar Z_i\longmapsto\bar X_{p(i)},\qquad
\bar X_i\longmapsto\bar Z_{p(i)}.
\]

The two physical translation generators induce the commuting logical
permutations

\[
T_x=(0\;2\;4\;6)(1\;3\;5\;7),\qquad
T_y=(0\;1)(2\;3)(4\;5)(6\;7).
\]

Thus the logical indices form a regular `C_4 x C_2` orbit.

## Explicit physical automorphisms

Let $U_P$ denote the unitary that moves the quantum state at physical data
site $q_j$ to site $q_{P(j)}$.  The convention used here is

\[
U_P Z_{q_j}U_P^\dagger=Z_{q_{P(j)}},\qquad
U_P X_{q_j}U_P^\dagger=X_{q_{P(j)}}.
\]

### Physical lift of the logical `C_4` generator

The physical permutation $P_x$ is

\[
\begin{aligned}
P_x&=(q_{0}\;q_{1}\;q_{2}\;q_{3}\;q_{4}\;q_{5}\;q_{6}\;q_{7})(q_{8}\;q_{9}\;q_{10}\;q_{11}\;q_{12}\;q_{13}\;q_{14}\;q_{15}),\\&\quad (q_{16}\;q_{17}\;q_{18}\;q_{19}\;q_{20}\;q_{21}\;q_{22}\;q_{23})(q_{24}\;q_{25}\;q_{26}\;q_{27}\;q_{28}\;q_{29}\;q_{30}\;q_{31}),\\&\quad (q_{32}\;q_{33}\;q_{34}\;q_{35}\;q_{36}\;q_{37}\;q_{38}\;q_{39})(q_{40}\;q_{41}\;q_{42}\;q_{43}\;q_{44}\;q_{45}\;q_{46}\;q_{47}),\\&\quad (q_{48}\;q_{49}\;q_{50}\;q_{51}\;q_{52}\;q_{53}\;q_{54}\;q_{55})(q_{56}\;q_{57}\;q_{58}\;q_{59}\;q_{60}\;q_{61}\;q_{62}\;q_{63})
\end{aligned}
\]

Equivalently, on the strict planar `C_8 x C_4` data array,

\[
P_x:\quad L(u,v)\mapsto L(u+1,v),\qquad
R(u,v)\mapsto R(u+1,v).
\]

Its physical order is eight, although its induced logical order is four.  The
kernel element $P_x^4$ shifts every atom by four planar columns while acting
trivially on the logical subsystem.

### Physical lift of the logical `C_2` generator

The physical permutation $P_y$ is the following set of 32 simultaneous swaps:

\[
\begin{aligned}
P_y&=(q_{0}\;q_{32})(q_{1}\;q_{33})(q_{2}\;q_{34})(q_{3}\;q_{35}),\\&\quad (q_{4}\;q_{36})(q_{5}\;q_{37})(q_{6}\;q_{38})(q_{7}\;q_{39}),\\&\quad (q_{8}\;q_{60})(q_{9}\;q_{61})(q_{10}\;q_{62})(q_{11}\;q_{63}),\\&\quad (q_{12}\;q_{56})(q_{13}\;q_{57})(q_{14}\;q_{58})(q_{15}\;q_{59}),\\&\quad (q_{16}\;q_{52})(q_{17}\;q_{53})(q_{18}\;q_{54})(q_{19}\;q_{55}),\\&\quad (q_{20}\;q_{48})(q_{21}\;q_{49})(q_{22}\;q_{50})(q_{23}\;q_{51}),\\&\quad (q_{24}\;q_{44})(q_{25}\;q_{45})(q_{26}\;q_{46})(q_{27}\;q_{47}),\\&\quad (q_{28}\;q_{40})(q_{29}\;q_{41})(q_{30}\;q_{42})(q_{31}\;q_{43})
\end{aligned}
\]

In strict planar coordinates its endpoint map is

\[
P_y:\quad L(u,v)\mapsto R(u-2v,-v),\qquad
R(u,v)\mapsto L(u-2v,-v),
\]

where the first coordinate is modulo eight and the second is modulo four.
Thus $P_y$ exchanges the two BB data halves, reflects the row coordinate, and
applies a row-dependent horizontal displacement.  Its physical and logical
orders are both two.

The physical generators commute.  They generate a physical
$C_8\times C_2$ action whose logical quotient by the logically trivial
$\langle P_x^4\rangle$ action is the desired $C_4\times C_2$ translation
group.

Both permutations map each displayed X-check and Z-check row directly to
another displayed row, rather than merely preserving the check row space.  If
$s_a^X$ and $s_a^Z$ denote displayed check rows with $a=0,\ldots,31$, their
common row-label permutations are

\[
\begin{aligned}
r_x&=(0\;1\;2\;3\;4\;5\;6\;7)(8\;9\;10\;11\;12\;13\;14\;15),\\&\quad (16\;17\;18\;19\;20\;21\;22\;23)(24\;25\;26\;27\;28\;29\;30\;31)\\[2pt]
r_y&=(8\;28)(9\;29)(10\;30)(11\;31),\\&\quad (12\;24)(13\;25)(14\;26)(15\;27),\\&\quad (16\;20)(17\;21)(18\;22)(19\;23).
\end{aligned}
\]

Fixed check-row labels are omitted from this cycle notation; in particular,
$r_y$ fixes rows $0,\ldots,7$.

These equations specify the endpoint permutations exactly.  They certify code
automorphisms, but they do not by themselves provide collision-free continuous
optical-tweezer trajectories between the endpoints.

## Simultaneous X/Z syndrome schedule

There are 32 X ancillas $x_0,\ldots,x_{31}$ and 32 Z ancillas
$z_0,\ldots,z_{31}$.  For each syndrome round:

1. prepare every $x_a$ in $|+\rangle$ and every $z_a$ in $|0\rangle$;
2. execute the eight CNOT layers below in order;
3. measure every $x_a$ in the X basis and every $z_a$ in the Z basis.

The arrow gives the CNOT control-to-target direction:

\[
x_a\to q_j\quad\text{for an X-check interaction},\qquad
q_j\to z_a\quad\text{for a Z-check interaction}.
\]

Every layer contains 32 X-check CNOTs and 32 Z-check CNOTs.  Each data qubit
appears exactly once in each layer, so all 64 interactions occur in parallel.

### CNOT layer 0

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{0},\quad x_{1}\!\to q_{1},\quad x_{2}\!\to q_{2},\quad x_{3}\!\to q_{3},\quad x_{4}\!\to q_{4},\quad x_{5}\!\to q_{5},\quad x_{6}\!\to q_{6},\quad x_{7}\!\to q_{7},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{8},\quad x_{9}\!\to q_{9},\quad x_{10}\!\to q_{10},\quad x_{11}\!\to q_{11},\quad x_{12}\!\to q_{12},\quad x_{13}\!\to q_{13},\quad x_{14}\!\to q_{14},\quad x_{15}\!\to q_{15},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{16},\quad x_{17}\!\to q_{17},\quad x_{18}\!\to q_{18},\quad x_{19}\!\to q_{19},\quad x_{20}\!\to q_{20},\quad x_{21}\!\to q_{21},\quad x_{22}\!\to q_{22},\quad x_{23}\!\to q_{23},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{24},\quad x_{25}\!\to q_{25},\quad x_{26}\!\to q_{26},\quad x_{27}\!\to q_{27},\quad x_{28}\!\to q_{28},\quad x_{29}\!\to q_{29},\quad x_{30}\!\to q_{30},\quad x_{31}\!\to q_{31},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{33}\!\to z_{0},\quad q_{34}\!\to z_{1},\quad q_{35}\!\to z_{2},\quad q_{36}\!\to z_{3},\quad q_{37}\!\to z_{4},\quad q_{38}\!\to z_{5},\quad q_{39}\!\to z_{6},\quad q_{32}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{41}\!\to z_{8},\quad q_{42}\!\to z_{9},\quad q_{43}\!\to z_{10},\quad q_{44}\!\to z_{11},\quad q_{45}\!\to z_{12},\quad q_{46}\!\to z_{13},\quad q_{47}\!\to z_{14},\quad q_{40}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{49}\!\to z_{16},\quad q_{50}\!\to z_{17},\quad q_{51}\!\to z_{18},\quad q_{52}\!\to z_{19},\quad q_{53}\!\to z_{20},\quad q_{54}\!\to z_{21},\quad q_{55}\!\to z_{22},\quad q_{48}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{57}\!\to z_{24},\quad q_{58}\!\to z_{25},\quad q_{59}\!\to z_{26},\quad q_{60}\!\to z_{27},\quad q_{61}\!\to z_{28},\quad q_{62}\!\to z_{29},\quad q_{63}\!\to z_{30},\quad q_{56}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 1

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{60},\quad x_{1}\!\to q_{61},\quad x_{2}\!\to q_{62},\quad x_{3}\!\to q_{63},\quad x_{4}\!\to q_{56},\quad x_{5}\!\to q_{57},\quad x_{6}\!\to q_{58},\quad x_{7}\!\to q_{59},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{32},\quad x_{9}\!\to q_{33},\quad x_{10}\!\to q_{34},\quad x_{11}\!\to q_{35},\quad x_{12}\!\to q_{36},\quad x_{13}\!\to q_{37},\quad x_{14}\!\to q_{38},\quad x_{15}\!\to q_{39},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{40},\quad x_{17}\!\to q_{41},\quad x_{18}\!\to q_{42},\quad x_{19}\!\to q_{43},\quad x_{20}\!\to q_{44},\quad x_{21}\!\to q_{45},\quad x_{22}\!\to q_{46},\quad x_{23}\!\to q_{47},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{48},\quad x_{25}\!\to q_{49},\quad x_{26}\!\to q_{50},\quad x_{27}\!\to q_{51},\quad x_{28}\!\to q_{52},\quad x_{29}\!\to q_{53},\quad x_{30}\!\to q_{54},\quad x_{31}\!\to q_{55},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{7}\!\to z_{0},\quad q_{0}\!\to z_{1},\quad q_{1}\!\to z_{2},\quad q_{2}\!\to z_{3},\quad q_{3}\!\to z_{4},\quad q_{4}\!\to z_{5},\quad q_{5}\!\to z_{6},\quad q_{6}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{15}\!\to z_{8},\quad q_{8}\!\to z_{9},\quad q_{9}\!\to z_{10},\quad q_{10}\!\to z_{11},\quad q_{11}\!\to z_{12},\quad q_{12}\!\to z_{13},\quad q_{13}\!\to z_{14},\quad q_{14}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{23}\!\to z_{16},\quad q_{16}\!\to z_{17},\quad q_{17}\!\to z_{18},\quad q_{18}\!\to z_{19},\quad q_{19}\!\to z_{20},\quad q_{20}\!\to z_{21},\quad q_{21}\!\to z_{22},\quad q_{22}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{31}\!\to z_{24},\quad q_{24}\!\to z_{25},\quad q_{25}\!\to z_{26},\quad q_{26}\!\to z_{27},\quad q_{27}\!\to z_{28},\quad q_{28}\!\to z_{29},\quad q_{29}\!\to z_{30},\quad q_{30}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 2

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{1},\quad x_{1}\!\to q_{2},\quad x_{2}\!\to q_{3},\quad x_{3}\!\to q_{4},\quad x_{4}\!\to q_{5},\quad x_{5}\!\to q_{6},\quad x_{6}\!\to q_{7},\quad x_{7}\!\to q_{0},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{9},\quad x_{9}\!\to q_{10},\quad x_{10}\!\to q_{11},\quad x_{11}\!\to q_{12},\quad x_{12}\!\to q_{13},\quad x_{13}\!\to q_{14},\quad x_{14}\!\to q_{15},\quad x_{15}\!\to q_{8},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{17},\quad x_{17}\!\to q_{18},\quad x_{18}\!\to q_{19},\quad x_{19}\!\to q_{20},\quad x_{20}\!\to q_{21},\quad x_{21}\!\to q_{22},\quad x_{22}\!\to q_{23},\quad x_{23}\!\to q_{16},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{25},\quad x_{25}\!\to q_{26},\quad x_{26}\!\to q_{27},\quad x_{27}\!\to q_{28},\quad x_{28}\!\to q_{29},\quad x_{29}\!\to q_{30},\quad x_{30}\!\to q_{31},\quad x_{31}\!\to q_{24},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{39}\!\to z_{0},\quad q_{32}\!\to z_{1},\quad q_{33}\!\to z_{2},\quad q_{34}\!\to z_{3},\quad q_{35}\!\to z_{4},\quad q_{36}\!\to z_{5},\quad q_{37}\!\to z_{6},\quad q_{38}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{47}\!\to z_{8},\quad q_{40}\!\to z_{9},\quad q_{41}\!\to z_{10},\quad q_{42}\!\to z_{11},\quad q_{43}\!\to z_{12},\quad q_{44}\!\to z_{13},\quad q_{45}\!\to z_{14},\quad q_{46}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{55}\!\to z_{16},\quad q_{48}\!\to z_{17},\quad q_{49}\!\to z_{18},\quad q_{50}\!\to z_{19},\quad q_{51}\!\to z_{20},\quad q_{52}\!\to z_{21},\quad q_{53}\!\to z_{22},\quad q_{54}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{63}\!\to z_{24},\quad q_{56}\!\to z_{25},\quad q_{57}\!\to z_{26},\quad q_{58}\!\to z_{27},\quad q_{59}\!\to z_{28},\quad q_{60}\!\to z_{29},\quad q_{61}\!\to z_{30},\quad q_{62}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 3

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{8},\quad x_{1}\!\to q_{9},\quad x_{2}\!\to q_{10},\quad x_{3}\!\to q_{11},\quad x_{4}\!\to q_{12},\quad x_{5}\!\to q_{13},\quad x_{6}\!\to q_{14},\quad x_{7}\!\to q_{15},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{16},\quad x_{9}\!\to q_{17},\quad x_{10}\!\to q_{18},\quad x_{11}\!\to q_{19},\quad x_{12}\!\to q_{20},\quad x_{13}\!\to q_{21},\quad x_{14}\!\to q_{22},\quad x_{15}\!\to q_{23},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{24},\quad x_{17}\!\to q_{25},\quad x_{18}\!\to q_{26},\quad x_{19}\!\to q_{27},\quad x_{20}\!\to q_{28},\quad x_{21}\!\to q_{29},\quad x_{22}\!\to q_{30},\quad x_{23}\!\to q_{31},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{4},\quad x_{25}\!\to q_{5},\quad x_{26}\!\to q_{6},\quad x_{27}\!\to q_{7},\quad x_{28}\!\to q_{0},\quad x_{29}\!\to q_{1},\quad x_{30}\!\to q_{2},\quad x_{31}\!\to q_{3},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{32}\!\to z_{0},\quad q_{33}\!\to z_{1},\quad q_{34}\!\to z_{2},\quad q_{35}\!\to z_{3},\quad q_{36}\!\to z_{4},\quad q_{37}\!\to z_{5},\quad q_{38}\!\to z_{6},\quad q_{39}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{40}\!\to z_{8},\quad q_{41}\!\to z_{9},\quad q_{42}\!\to z_{10},\quad q_{43}\!\to z_{11},\quad q_{44}\!\to z_{12},\quad q_{45}\!\to z_{13},\quad q_{46}\!\to z_{14},\quad q_{47}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{48}\!\to z_{16},\quad q_{49}\!\to z_{17},\quad q_{50}\!\to z_{18},\quad q_{51}\!\to z_{19},\quad q_{52}\!\to z_{20},\quad q_{53}\!\to z_{21},\quad q_{54}\!\to z_{22},\quad q_{55}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{56}\!\to z_{24},\quad q_{57}\!\to z_{25},\quad q_{58}\!\to z_{26},\quad q_{59}\!\to z_{27},\quad q_{60}\!\to z_{28},\quad q_{61}\!\to z_{29},\quad q_{62}\!\to z_{30},\quad q_{63}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 4

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{32},\quad x_{1}\!\to q_{33},\quad x_{2}\!\to q_{34},\quad x_{3}\!\to q_{35},\quad x_{4}\!\to q_{36},\quad x_{5}\!\to q_{37},\quad x_{6}\!\to q_{38},\quad x_{7}\!\to q_{39},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{40},\quad x_{9}\!\to q_{41},\quad x_{10}\!\to q_{42},\quad x_{11}\!\to q_{43},\quad x_{12}\!\to q_{44},\quad x_{13}\!\to q_{45},\quad x_{14}\!\to q_{46},\quad x_{15}\!\to q_{47},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{48},\quad x_{17}\!\to q_{49},\quad x_{18}\!\to q_{50},\quad x_{19}\!\to q_{51},\quad x_{20}\!\to q_{52},\quad x_{21}\!\to q_{53},\quad x_{22}\!\to q_{54},\quad x_{23}\!\to q_{55},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{56},\quad x_{25}\!\to q_{57},\quad x_{26}\!\to q_{58},\quad x_{27}\!\to q_{59},\quad x_{28}\!\to q_{60},\quad x_{29}\!\to q_{61},\quad x_{30}\!\to q_{62},\quad x_{31}\!\to q_{63},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{8}\!\to z_{0},\quad q_{9}\!\to z_{1},\quad q_{10}\!\to z_{2},\quad q_{11}\!\to z_{3},\quad q_{12}\!\to z_{4},\quad q_{13}\!\to z_{5},\quad q_{14}\!\to z_{6},\quad q_{15}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{16}\!\to z_{8},\quad q_{17}\!\to z_{9},\quad q_{18}\!\to z_{10},\quad q_{19}\!\to z_{11},\quad q_{20}\!\to z_{12},\quad q_{21}\!\to z_{13},\quad q_{22}\!\to z_{14},\quad q_{23}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{24}\!\to z_{16},\quad q_{25}\!\to z_{17},\quad q_{26}\!\to z_{18},\quad q_{27}\!\to z_{19},\quad q_{28}\!\to z_{20},\quad q_{29}\!\to z_{21},\quad q_{30}\!\to z_{22},\quad q_{31}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{4}\!\to z_{24},\quad q_{5}\!\to z_{25},\quad q_{6}\!\to z_{26},\quad q_{7}\!\to z_{27},\quad q_{0}\!\to z_{28},\quad q_{1}\!\to z_{29},\quad q_{2}\!\to z_{30},\quad q_{3}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 5

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{39},\quad x_{1}\!\to q_{32},\quad x_{2}\!\to q_{33},\quad x_{3}\!\to q_{34},\quad x_{4}\!\to q_{35},\quad x_{5}\!\to q_{36},\quad x_{6}\!\to q_{37},\quad x_{7}\!\to q_{38},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{47},\quad x_{9}\!\to q_{40},\quad x_{10}\!\to q_{41},\quad x_{11}\!\to q_{42},\quad x_{12}\!\to q_{43},\quad x_{13}\!\to q_{44},\quad x_{14}\!\to q_{45},\quad x_{15}\!\to q_{46},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{55},\quad x_{17}\!\to q_{48},\quad x_{18}\!\to q_{49},\quad x_{19}\!\to q_{50},\quad x_{20}\!\to q_{51},\quad x_{21}\!\to q_{52},\quad x_{22}\!\to q_{53},\quad x_{23}\!\to q_{54},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{63},\quad x_{25}\!\to q_{56},\quad x_{26}\!\to q_{57},\quad x_{27}\!\to q_{58},\quad x_{28}\!\to q_{59},\quad x_{29}\!\to q_{60},\quad x_{30}\!\to q_{61},\quad x_{31}\!\to q_{62},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{1}\!\to z_{0},\quad q_{2}\!\to z_{1},\quad q_{3}\!\to z_{2},\quad q_{4}\!\to z_{3},\quad q_{5}\!\to z_{4},\quad q_{6}\!\to z_{5},\quad q_{7}\!\to z_{6},\quad q_{0}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{9}\!\to z_{8},\quad q_{10}\!\to z_{9},\quad q_{11}\!\to z_{10},\quad q_{12}\!\to z_{11},\quad q_{13}\!\to z_{12},\quad q_{14}\!\to z_{13},\quad q_{15}\!\to z_{14},\quad q_{8}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{17}\!\to z_{16},\quad q_{18}\!\to z_{17},\quad q_{19}\!\to z_{18},\quad q_{20}\!\to z_{19},\quad q_{21}\!\to z_{20},\quad q_{22}\!\to z_{21},\quad q_{23}\!\to z_{22},\quad q_{16}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{25}\!\to z_{24},\quad q_{26}\!\to z_{25},\quad q_{27}\!\to z_{26},\quad q_{28}\!\to z_{27},\quad q_{29}\!\to z_{28},\quad q_{30}\!\to z_{29},\quad q_{31}\!\to z_{30},\quad q_{24}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 6

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{7},\quad x_{1}\!\to q_{0},\quad x_{2}\!\to q_{1},\quad x_{3}\!\to q_{2},\quad x_{4}\!\to q_{3},\quad x_{5}\!\to q_{4},\quad x_{6}\!\to q_{5},\quad x_{7}\!\to q_{6},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{15},\quad x_{9}\!\to q_{8},\quad x_{10}\!\to q_{9},\quad x_{11}\!\to q_{10},\quad x_{12}\!\to q_{11},\quad x_{13}\!\to q_{12},\quad x_{14}\!\to q_{13},\quad x_{15}\!\to q_{14},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{23},\quad x_{17}\!\to q_{16},\quad x_{18}\!\to q_{17},\quad x_{19}\!\to q_{18},\quad x_{20}\!\to q_{19},\quad x_{21}\!\to q_{20},\quad x_{22}\!\to q_{21},\quad x_{23}\!\to q_{22},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{31},\quad x_{25}\!\to q_{24},\quad x_{26}\!\to q_{25},\quad x_{27}\!\to q_{26},\quad x_{28}\!\to q_{27},\quad x_{29}\!\to q_{28},\quad x_{30}\!\to q_{29},\quad x_{31}\!\to q_{30},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{60}\!\to z_{0},\quad q_{61}\!\to z_{1},\quad q_{62}\!\to z_{2},\quad q_{63}\!\to z_{3},\quad q_{56}\!\to z_{4},\quad q_{57}\!\to z_{5},\quad q_{58}\!\to z_{6},\quad q_{59}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{32}\!\to z_{8},\quad q_{33}\!\to z_{9},\quad q_{34}\!\to z_{10},\quad q_{35}\!\to z_{11},\quad q_{36}\!\to z_{12},\quad q_{37}\!\to z_{13},\quad q_{38}\!\to z_{14},\quad q_{39}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{40}\!\to z_{16},\quad q_{41}\!\to z_{17},\quad q_{42}\!\to z_{18},\quad q_{43}\!\to z_{19},\quad q_{44}\!\to z_{20},\quad q_{45}\!\to z_{21},\quad q_{46}\!\to z_{22},\quad q_{47}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{48}\!\to z_{24},\quad q_{49}\!\to z_{25},\quad q_{50}\!\to z_{26},\quad q_{51}\!\to z_{27},\quad q_{52}\!\to z_{28},\quad q_{53}\!\to z_{29},\quad q_{54}\!\to z_{30},\quad q_{55}\!\to z_{31}.
\end{aligned}
\]

### CNOT layer 7

\[
\begin{aligned}
\mathrm{X}:\quad &x_{0}\!\to q_{33},\quad x_{1}\!\to q_{34},\quad x_{2}\!\to q_{35},\quad x_{3}\!\to q_{36},\quad x_{4}\!\to q_{37},\quad x_{5}\!\to q_{38},\quad x_{6}\!\to q_{39},\quad x_{7}\!\to q_{32},\\
\phantom{\mathrm{X}:\quad}&x_{8}\!\to q_{41},\quad x_{9}\!\to q_{42},\quad x_{10}\!\to q_{43},\quad x_{11}\!\to q_{44},\quad x_{12}\!\to q_{45},\quad x_{13}\!\to q_{46},\quad x_{14}\!\to q_{47},\quad x_{15}\!\to q_{40},\\
\phantom{\mathrm{X}:\quad}&x_{16}\!\to q_{49},\quad x_{17}\!\to q_{50},\quad x_{18}\!\to q_{51},\quad x_{19}\!\to q_{52},\quad x_{20}\!\to q_{53},\quad x_{21}\!\to q_{54},\quad x_{22}\!\to q_{55},\quad x_{23}\!\to q_{48},\\
\phantom{\mathrm{X}:\quad}&x_{24}\!\to q_{57},\quad x_{25}\!\to q_{58},\quad x_{26}\!\to q_{59},\quad x_{27}\!\to q_{60},\quad x_{28}\!\to q_{61},\quad x_{29}\!\to q_{62},\quad x_{30}\!\to q_{63},\quad x_{31}\!\to q_{56},\\
\\[-2pt]
\mathrm{Z}:\quad &q_{0}\!\to z_{0},\quad q_{1}\!\to z_{1},\quad q_{2}\!\to z_{2},\quad q_{3}\!\to z_{3},\quad q_{4}\!\to z_{4},\quad q_{5}\!\to z_{5},\quad q_{6}\!\to z_{6},\quad q_{7}\!\to z_{7},\\
\phantom{\mathrm{X}:\quad}&q_{8}\!\to z_{8},\quad q_{9}\!\to z_{9},\quad q_{10}\!\to z_{10},\quad q_{11}\!\to z_{11},\quad q_{12}\!\to z_{12},\quad q_{13}\!\to z_{13},\quad q_{14}\!\to z_{14},\quad q_{15}\!\to z_{15},\\
\phantom{\mathrm{X}:\quad}&q_{16}\!\to z_{16},\quad q_{17}\!\to z_{17},\quad q_{18}\!\to z_{18},\quad q_{19}\!\to z_{19},\quad q_{20}\!\to z_{20},\quad q_{21}\!\to z_{21},\quad q_{22}\!\to z_{22},\quad q_{23}\!\to z_{23},\\
\phantom{\mathrm{X}:\quad}&q_{24}\!\to z_{24},\quad q_{25}\!\to z_{25},\quad q_{26}\!\to z_{26},\quad q_{27}\!\to z_{27},\quad q_{28}\!\to z_{28},\quad q_{29}\!\to z_{29},\quad q_{30}\!\to z_{30},\quad q_{31}\!\to z_{31}.
\end{aligned}
\]

## Schedule interpretation

Across all eight layers, ancilla $x_a$ touches the eight data qubits in X-check
row $a$, while ancilla $z_a$ touches the eight data qubits in Z-check row $a$.
The X and Z row supports are identical because $H_X=H_Z$, but their CNOT
directions and their layer orders differ.  The displayed ordering is the
translation-symmetric, collision-free schedule with clean cross-ancilla
backaction and schedule ID `67e327dc7cb35335`.
