# cython: profile=True
# cython: binding=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
# # cython: overflowcheck=False
# # cython: cdivision=True
# # cython: boundscheck=False
# # cython: wraparound=False


cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as ssp
from functools import lru_cache


cdef extern from "gsl/gsl_sf.h":
  double gsl_sf_choose(unsigned int x, unsigned int n)


cdef unsigned int ncr(unsigned int x, unsigned int n):
    """ Fast N choose R. GSL wrapper. """
    if x >= n:
        return <unsigned int> gsl_sf_choose(x, n)
    else:
        return 0

cdef unsigned int ctz(unsigned int v):
    """ Returns the number of trailing zeros of v """
    cdef extern int __builtin_ctz(unsigned int x)
    return __builtin_ctz(v)


cdef unsigned int next_bit_perm(unsigned int v):
    """ Return the next lexicographical permutation of v """
    cdef unsigned int t;
    cdef extern int __builtin_ctz(unsigned int x)

    t = v | (v-1)
    return (t+1) | (((~t & -~t) -1) >> (__builtin_ctz(v)+1))


cdef np.ndarray[np.int_t, ndim=1] all_bit_perms(unsigned int v, unsigned int len_v):
    """ """
    cdef size_t i;
    cdef np.ndarray[np.int_t, ndim=1] ms = np.zeros(len_v, dtype=np.int)
    ms[0] = v
    for i in range(1, len_v):
        ms[i] = next_bit_perm(ms[i-1])
    return ms


cdef np.ndarray[np.int_t, ndim=1] one_age_idx_to_state(unsigned int si, unsigned int nbars, unsigned int nstars):
    """ """
    cdef:
        np.ndarray[np.int_t, ndim=1] I = np.empty(nbars+1, dtype=np.int)
        unsigned int csumI = 0
        extern int __builtin_ctz(unsigned int x)
        size_t i

    si += 2**(nbars+nstars)  # -1

    for i in range(nbars+1):
        # Count stars (zeroes) except previously counted and flipped bars
        I[i] = __builtin_ctz(si)-i-csumI
        csumI += I[i]
        si -= (si&-si)  # Flip least significant bit
    return I


cpdef barsize(unsigned int A, unsigned int R, np.ndarray[np.int_t, ndim=1] N):
    """ Number of states for A demographic levels (with populations according
        to N), R infection levels.
    """
    cdef unsigned int ns
    cdef size_t i, a
    cdef unsigned int len_v
    cdef unsigned int size = 1

    for i, a in enumerate(range(1, A+1)):
        ns = N[i]
        len_v = ncr(R+ns, R)
        size *= len_v

    return size


@cython.embedsignature(True)
cpdef np.ndarray[np.int_t, ndim=1] idx_to_state(unsigned int j, unsigned int A, unsigned int R, np.ndarray[np.int_t, ndim=1] N):
    """


    Args:
        j
    """
    cdef:
        np.ndarray[np.int_t, ndim=1] tmp_a = np.ones(A, dtype=np.int)
        np.ndarray[np.int_t, ndim=1] STATE = np.empty(A*R, dtype=np.int)
        np.ndarray[np.int_t, ndim=1] ms
        unsigned int ns, v0, subsize, len_v
        size_t i, a, b

    for i, a in enumerate(range(1, A+1)):
        ns = N[i]
        v0 = 2**R - 1  # ns 0's and R 1's - TODO could be factored into all_bit_perms?

        len_v = ncr(R+ns, R)  # TODO could be factored into all_bit_perms?
        ms = all_bit_perms(v0, len_v)
        d = {k: v for k, v in enumerate(ms)}

        for b in range(0, A-a):
            tmp_a[i] *= ncr(N[A-b-1]+R, R)
        tmp_a[i] = int(tmp_a[i])

        subsize = ms.size
        STATE[R*(A-1-i):R*(A-i)] = one_age_idx_to_state(d[(j//tmp_a[i])%(subsize)], R, ns)[:-1]
    return STATE[::-1]


@cython.embedsignature(True)
cpdef unsigned int el_f_gen(
    np.ndarray[np.int_t, ndim=1] N,
    np.ndarray[np.int_t, ndim=1] I
):
    """ Cython method to find markov chain index corresponding to infectious state I

    Args:
        N (np.ndarray[int]): Array of numbers representing population in each
            demographic bracket.
        I (np.ndarray[int]): Array representing infection levels.

    Returns:
        int
    """
    return el_f_gen_base(N, I)


@lru_cache()
def el_f_gen_memo(np.ndarray[np.int_t, ndim=1] N, np.ndarray[np.int_t, ndim=1] I):
    """ Memoized version of el_f_gen.

    Find markov chain index corresponding to infectious state I.
    If memoization is not required, `el_f_gen` should be faster.

    Args:
        N (np.ndarray[int]): Array of numbers representing population in each
            demographic bracket.
        I (np.ndarray[int]): Array representing infection levels.

    Returns:
        int
    """
    return el_f_gen_base(N, I)


cdef unsigned int el_f_gen_base(np.ndarray[np.int_t, ndim=1] N, np.ndarray[np.int_t, ndim=1] I):
    # Internal def
    cdef unsigned int A = N.shape[0]
    cdef unsigned int R = I.shape[0] // A
    cdef unsigned int c1, c2, c3, p
    cdef size_t a, r, i, b

    c3 = 0
    for a in range(1, A+1):
        c2 = 0
        for r in range(1, R+1):
            c1 = 0
            for i in range(R-r+1, R+1):
                c1 += I[R*(a-1)+(i-1)]
            c2 += ncr(c1+r-1, r)

        p = 1
        for b in range(0, A-a):
            p *= ncr(N[A-b-1]+R, R)

        c3 += c2*p
    return c3


cdef np.ndarray[np.int_t, ndim=1] f(unsigned int R, unsigned int A, unsigned int i, unsigned int a):
    """ Increase 1 in (r-1, a)th state and decrease 1 in (r, a)th state """
    cdef np.ndarray[np.int_t, ndim=1] s = np.zeros(A*R, dtype=np.int)
    if i > 0:
        s[a*R+(i-1)] = -1
    s[a*R+i] = 1
    return s


def SIS_AR_gen(unsigned int A, unsigned int R, np.ndarray[np.int_t, ndim=1] Ntup,
                 np.ndarray[np.double_t, ndim=2] beta,
                 np.ndarray[np.double_t, ndim=2] gamma,
                 np.ndarray[np.double_t, ndim=2] rho,
                 double eps, double alpha, unsigned int cf):
    """ Return sparse transition matrix for a given model specification

    Args:
        A :
        R :
        Ntup :
        beta :
        gamma :
        rho :
        eps :
        alpha :
        cf :

    Returns:
        scipy.sparse.csr_matrix
    """
    # X{r, a} = risk level r & age class a
    cdef:
        unsigned int SIZE = barsize(A, R, Ntup)  # Number of states in Markov Chain
        unsigned int N = Ntup.sum()  # Total population
        np.ndarray[np.double_t, ndim=1] D = np.zeros(cf*SIZE, dtype=np.double)
        np.ndarray[np.int_t, ndim=1] r = np.zeros(cf*SIZE, dtype=np.int)
        np.ndarray[np.int_t, ndim=1] c = np.zeros(cf*SIZE, dtype=np.int)

        unsigned int i = 0
        np.ndarray[np.int_t, ndim=2] X = np.empty((R+1, A), dtype=np.int)
        np.ndarray[np.int_t, ndim=1] Xf = np.empty(R+1*A, dtype=np.int)
        np.ndarray[np.int_t, ndim=1] Xi = np.empty(R+1, dtype=np.int)
        unsigned int I
        double Res, tmp
        size_t a, ri
        unsigned int Xij

    beta = beta * 1/(<double> N-1)**alpha

    for stidx in range(SIZE):
        Xf = idx_to_state(stidx, A, R, Ntup)
        X[1:, :] = Xf.reshape((A, R)).T
        X[0, :] = Ntup-X[1:, :].sum(axis=0)
        I = el_f_gen(Ntup, Xf)
        Res = (beta * X).sum() + eps
        tmp = 0
        for a, Xi in enumerate(X.T):
            if Xi[0] >= 1:
                r[i] = I
                c[i] = el_f_gen(Ntup, Xf+f(R, A, 0, a))
                D[i] = Res*Xi[0]*rho[0, a] # S[a] -> I[1,a]
                tmp += D[i]
                i += 1
            for ri, Xij in enumerate(Xi[1:-1]):
                if Xij >= 1:
                    r[i] = I
                    c[i] = el_f_gen(Ntup, Xf+f(R, A, ri+1, a))
                    D[i] = Res*Xij*rho[ri+1, a]  # X[ri+1,a] -> X[ri+2,a]
                    tmp += D[i]
                    i += 1
                    r[i] = I
                    c[i] = el_f_gen(Ntup, Xf-f(R, A, ri, a))
                    D[i] = gamma[ri+1, a]*Xij  # X[ri+1, a] -> X[ri, a]
                    tmp += D[i]
                    i += 1
            if Xi[-1] >= 1:
                r[i] = I
                c[i] = el_f_gen(Ntup, Xf-f(R, A, R-1, a))
                D[i] = gamma[-1, a]*Xi[-1]  # I[R, a] - I[R-1, a]
                tmp += D[i]
                i += 1
        r[i] = I
        c[i] = I
        D[i] = -tmp
        i += 1

    return ssp.coo_matrix((D, (r, c)), shape=(SIZE, SIZE)).tocsr().T

