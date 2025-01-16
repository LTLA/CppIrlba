# Thoughts on Gram-Schmidt

Currently, each vector from the Lanczos process is orthogonalized against the existing vectors using classical Gram-Schmidt (CGS).
However, most of the literature suggests using modified Gram-Schmidt (MGS) instead, as this is more numerically stable.
I was wondering whether we needed to switch our implementation in **irlba** to match this.

As it turns out (thanks Bryan), the vectors from the Lanczos process are theoretically orthonormal but are subject to numerical instability.
The CGS step is just performed at each iteration of the process to eliminate the orthogonality error from this instability.
Some examination of the CGS error (derived from [this link](https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html)) indicates that it is proportional to the dot-product of the original vectors.
Paraphrased:

1. Let's define the Lanczos vectors $v_1, v_2, ..., v_k$, that have some small orthogonality error, e.g., $v_iv_j$ is close to zero for $i \ne j$.
2. As each Lanczos vector is produced, we use CGS to re-orthogonalize it against the existing (re-orthogonalized) vectors.
   This produces $w_1, w_2, ..., w_k$.
3. For $k = 1$, $w_1 = v_1 / \| v_1 \|$ as there's nothing to orthogonalize against.
4. For $k = 2$, $q_2 = v_2 - (w_1 \cdot v_2) w_1$ and then $w_2 = q_2 / \| q_2 \|$.
   Let's say that this calculation is not exact so we get $w_1 \cdot w_2 = \epsilon_{12}$ for some small numerical error.
5. For $k = 3$, $q_3 = v_3 - (w_1 \cdot v_3) w_1 - (w_2 \cdot v_3) w_2$ and then $w_3 = q_3 / \| q_3 \|$.
   This gives us $w_3 \cdot w_2 = (v_3 \cdot w_2 - (w_1 \cdot v_3) (w_1 \cdot w_2) - (w_2 \cdot v_3) (w_2 \cdot w_2)) \| q_3 \|^{-1}$,
   plus some numerical error that I won't bother including, which boils down to $- (v_1 \cdot v_3) \epsilon_{12} \| q_3 \|^{-1} \| q_1 \|^{-1}$.

In short, the orthogonalization error for each vector in CGS depends on the orthogonalization errors of prior vectors scaled by the dot product of the original vectors (here, $v_1 \cdot v_3$). 
If the dot product is small, e.g., because the vectors were near-orthogonal in the first place, then the CGS errors will also be low.
This justifies the continued use of CGS in **irlba**, which is a bit more efficient than MGS.
(Though to be honest, I don't think the orthogonalization is really in a hot loop anyway, as the multiplication takes most of the time, so it wouldn't really hurt to switch.)

To show that the above reasoning bears out in practice, here's a simulation:

```r
classical <- function(x) {
    for (i in 2:ncol(x)) {
        v <- x[,i]
        left <- x[,1:(i-1),drop=FALSE]
        proj <- crossprod(left, v) 
        v <- v - left %*% proj
        x[,i] <- v / sqrt(sum(v^2))
    }
    x
}

modified <- function(x) {
    for (i in 2:ncol(x)) {
        v <- x[,i]
        for (j in 1:(i-1)) {
            left <- x[,j]
            v <- v - sum(left * v) * left
        }
        x[,i] <- v / sqrt(sum(v^2))
    }
    x
}

assess_orthogonality <- function(x) { # i.e., Frobenius norm of off-diagonals
    out <- crossprod(x)
    diag(out) <- 0
    sqrt(sum(out^2))
}

# Near-orthogonal random vectors.
dim <- 2000
nvec <- 500
y <- matrix(rnorm(dim * nvec), ncol=nvec)
y <- t(t(y) / sqrt(colSums(y^2)))
assess_orthogonality(y)
## [1] 11.15392

cy <- classical(y)
assess_orthogonality(cy)
## [1] 2.277818e-14

my <- modified(y)
assess_orthogonality(my)
## [1] 1.224996e-14

# Very strong linear dependence.
z <- matrix(rnorm(dim * nvec, sd=0.01), ncol=nvec) + rnorm(dim)
z <- t(t(z) / sqrt(colSums(z^2)))
assess_orthogonality(z)
## [1] 499.4463

cz <- classical(z)
assess_orthogonality(cz)
## [1] 3.938994e-09

mz <- modified(z)
assess_orthogonality(mz)
## [1] 1.072133e-13
```

We can see that CGS has comparable orthogonality error to MGS when the input vectors are nearly orthogonal.
However, the former is several orders of magnitude worse when the input vectors exhibit strong linear dependencies. 
This is consistent with our formulation of the orthogonality error.
