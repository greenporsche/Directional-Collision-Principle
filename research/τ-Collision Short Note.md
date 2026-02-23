# A Quantitative Collision Principle for Odd Lipschitz Functions

## Abstract

We study real-valued functions on metric spaces equipped with an involutive symmetry. For continuous functions that are odd with respect to the involution, path-connectedness guarantees the existence of a zero. When additional Lipschitz regularity is assumed, we prove a sharp quantitative lower bound on the distance from any point to the zero set in terms of the function value and the Lipschitz constants. The result may be viewed as a metric refinement of intermediate value–type arguments and clarifies the geometric content of symmetry-forced degeneracies.

---

## 1. Introduction

Let ( (V,d) ) be a metric space and let
[
\tau:V\to V
]
be an involution, i.e. ( \tau\circ\tau=\mathrm{id}\_V ).
Functions that are _odd_ with respect to such a symmetry arise naturally in many contexts, and it is well known that oddness combined with connectedness forces the existence of zeros.

The purpose of this note is twofold:

1. To isolate a minimal, domain-free existence principle for such zeros.
2. To establish a **quantitative geometric bound** describing how far any point with nonzero value must lie from the zero set, assuming Lipschitz regularity.

Importantly, the quantitative result is one-sided: Lipschitz bounds provide **lower bounds** on the distance to the zero set, but cannot guarantee proximity of a zero without additional assumptions. This distinction appears to be underemphasized in informal discussions of symmetry-based arguments.

---

## 2. Setting and Definitions

Let ( (V,d) ) be a metric space.

### Definition 2.1 (Involution)

A map ( \tau:V\to V ) is called an **involution** if
[
\tau(\tau(v))=v \quad \text{for all } v\in V.
]

### Definition 2.2 (Odd function)

A function ( g:V\to\mathbb R ) is **odd with respect to ( \tau )** if
[
g(\tau(v))=-g(v)\quad\forall v\in V.
]

Given a function ( A:V\to\mathbb R ), define
[
g(v):=A(v)-A(\tau(v)).
]
Then ( g ) is automatically odd with respect to ( \tau ).

### Definition 2.3 (Collision set)

The **collision set** (or zero set) associated with ( g ) is
[
Z:={v\in V:\ g(v)=0}
={v\in V:\ A(v)=A(\tau(v))}.
]

---

## 3. Existence of Collision Points

We begin with a purely topological statement.

### Theorem 3.1 (Existence)

If ( V ) is path-connected and ( g:V\to\mathbb R ) is continuous and odd with respect to an involution ( \tau ), then
[
Z\neq\varnothing.
]

#### Proof

Let ( v\in V ). Since ( V ) is path-connected, there exists a continuous path
[
\gamma:[0,1]\to V
\quad\text{with}\quad
\gamma(0)=v,\ \gamma(1)=\tau(v).
]
By oddness,
[
g(\gamma(0))=g(v), \qquad
g(\gamma(1))=g(\tau(v))=-g(v).
]
Continuity of ( g\circ\gamma ) and the Intermediate Value Theorem yield ( t_0\in(0,1) ) such that ( g(\gamma(t_0))=0 ). ∎

### Remark 3.2

No metric structure is required for Theorem 3.1. In particular, when ( V=S^n ) and ( \tau ) is the antipodal map, the conclusion for real-valued functions follows from oddness and connectedness alone, without invoking the Borsuk–Ulam theorem.

---

## 4. Lipschitz Regularity and Quantitative Geometry

We now impose metric regularity and study the geometry of ( Z ).

### Assumption 4.1 (Lipschitz regularity)

Assume:

- ( A:V\to\mathbb R ) is ( L )-Lipschitz:
  [
  |A(x)-A(y)|\le L,d(x,y),
  ]
- ( \tau:V\to V ) is ( K )-Lipschitz:
  [
  d(\tau(x),\tau(y))\le K,d(x,y).
  ]

---

### Lemma 4.2 (Lipschitz constant of the odd function)

The function ( g=A-A\circ\tau ) is Lipschitz with
[
\operatorname{Lip}(g)\le L(1+K).
]

#### Proof

For any ( x,y\in V ),
[
\begin{aligned}
|g(x)-g(y)|
&\le |A(x)-A(y)| + |A(\tau(x))-A(\tau(y))| \
&\le Ld(x,y)+L,d(\tau(x),\tau(y)) \
&\le L(1+K)d(x,y).
\end{aligned}
]
∎

---

### Theorem 4.3 (Distance to the collision set)

For every ( v\in V ),
[
\boxed{
\operatorname{dist}(v,Z)
;\ge;
\frac{|A(v)-A(\tau(v))|}{L(1+K)}.
}
]

Equivalently, for every ( p\in Z ),
[
d(v,p)\ge \frac{|A(v)-A(\tau(v))|}{L(1+K)}.
]

#### Proof

Since ( g(p)=0 ) for all ( p\in Z ),
[
|g(v)|
=|g(v)-g(p)|
\le \operatorname{Lip}(g),d(v,p)
\le L(1+K)d(v,p).
]
Taking the infimum over ( p\in Z ) yields the claim. ∎

---

## 5. Local Consequences

### Corollary 5.1 (Collision-free balls)

Let ( v\in V ).
If ( g ) is ( C )-Lipschitz on the ball ( B(v,r) ) and
[
r<\frac{|g(v)|}{C},
]
then
[
Z\cap B(v,r)=\varnothing.
]

### Interpretation

Lipschitz regularity certifies a **minimum radius around any point with nonzero odd value that cannot contain a collision**.

---

## 6. Sharpness and Limitations

The bound in Theorem 4.3 is sharp in the sense that it cannot be reversed without additional assumptions. Lipschitz regularity acts as a **speed limit** on how fast ( g ) can decrease, but does not force a zero to occur nearby. Upper bounds on the distance to ( Z ) require extra geometric or coercivity conditions (e.g. inverse-Lipschitz behavior along a prescribed path).

---

## 7. Conclusion

Oddness with respect to an involution guarantees the existence of zeros on path-connected spaces. When combined with Lipschitz regularity, this symmetry yields a precise quantitative statement: the magnitude of asymmetry at a point lower-bounds its distance to the symmetry-enforced zero set. The result clarifies the geometric content of symmetry arguments and may be viewed as a metric refinement of classical intermediate value phenomena.

---
