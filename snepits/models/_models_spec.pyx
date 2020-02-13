# cython: profile=False
# cython: binding=True
# cython: linetrace=False
# distutils: define_macros=CYTHON_TRACE=0
## cython: language_level=3
# # cython: overflowcheck=False
# # cython: cdivision=True
# # cython: boundscheck=False
# # cython: wraparound=False
cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as ssp


cpdef unsigned int el_f(unsigned int Na, unsigned int Nc, unsigned int xl,
        unsigned int yl, unsigned int sa, unsigned int sc):
    return ((Nc + 1) * (Nc + 2) // 2 *
            ((Na - sa - xl) + ((Na - sa) * (Na - sa + 1) // 2)) +
            ((Nc - sc - yl) + ((Nc - sc) * (Nc - sc + 1) // 2)))

cpdef unsigned int el_f_A2_R3(unsigned int Na, unsigned int Nc, unsigned int xl, unsigned int xm, unsigned int yl, unsigned int ym, unsigned int sa, unsigned int sc):
    return ((Nc+1)*(Nc+2)*(Nc+3)//6 * (
        (Na-sa-xl-xm)+((Na-sa-xl)*(Na-sa-xl+1)//2)+((Na-sa)*(Na-sa+1)*(Na-sa+2)//6)
        ) + (
            (Nc-sc-yl-ym)+((Nc-sc-yl)*(Nc-sc-yl+1)//2)+((Nc-sc)*(Nc-sc+1)*(Nc-sc+2)//6)
            ))

cdef extern from "gsl/gsl_sf.h":
  double gsl_sf_choose(unsigned int x, unsigned int n)


cdef unsigned int ncr(unsigned int x, unsigned int n):
  if x >= n:
    return <unsigned int> gsl_sf_choose(x, n)
  else:
    return 0

cdef unsigned int ctz(unsigned int v):
    cdef extern int __builtin_ctz(unsigned int x)
    return __builtin_ctz(v)


cdef unsigned int next_bit_perm(unsigned int v):
    cdef unsigned int t;
    cdef extern int __builtin_ctz(unsigned int x)

    t = v | (v-1)
    return (t+1) | (((~t & -~t) -1) >> (__builtin_ctz(v)+1))


cdef np.ndarray[np.int_t, ndim=1] all_bit_perms(unsigned int v, unsigned int len_v):
    cdef size_t i;
    cdef np.ndarray[np.int_t, ndim=1] ms = np.zeros(len_v, dtype=np.int)
    ms[0] = v
    for i in range(1, len_v):
        ms[i] = next_bit_perm(ms[i-1])
    return ms


cdef np.ndarray[np.int_t, ndim=1] one_age_idx_to_state(unsigned int si, unsigned int nbars, unsigned int nstars):
    si += 2**(nbars+nstars)  # -1
    cdef np.ndarray[np.int_t, ndim=1] I = np.empty(nbars+1, dtype=np.int)
    cdef unsigned int csumI = 0
    cdef extern int __builtin_ctz(unsigned int x)
    cdef size_t i

    for i in range(nbars+1):
        # Count stars (zeroes) except previously counted and flipped bars
        I[i] = __builtin_ctz(si)-i-csumI
        csumI += I[i]
        si -= (si&-si)  # Flip least significant bit
    return I


cpdef barsize(unsigned int A, unsigned int R, np.ndarray[np.int_t, ndim=1] N):
    cdef unsigned int ns
    cdef size_t i, a
    cdef unsigned int len_v
    cdef unsigned int size = 1

    for i, a in enumerate(range(1, A+1)):
        ns = N[i]
        len_v = ncr(R+ns, R)
        size *= len_v

    return size


cpdef np.ndarray[np.int_t, ndim=1] idx_to_state(unsigned int j, unsigned int A, unsigned int R, np.ndarray[np.int_t, ndim=1] N):
    cdef np.ndarray[np.int_t, ndim=1] tmp_a = np.ones(A, dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] STATE = np.empty(A*R, dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] ms
    cdef unsigned int ns, v0, subsize
    cdef size_t i, a, b
    cdef unsigned int len_v;

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


cpdef SIS_ACR_sp_arr(
        unsigned int Na, unsigned int Nc, double bal, double bah, double bcl,
        double bch, double gal, double gah, double gcl, double gch, double eps,
        double rhoal, double rhoah, double rhocl, double rhoch):
    """
    Args:
        Na (uint): Number of adults
        Nc (uint): Number of children
        bal (double): Low risk adult transmission rate
        bah (double): High risk adult transmission rate
        bcl (double): Low risk child transmission rate
        bch (double): High risk child transmission rate
        gal (double): Low risk adult recovery rate
        gah (double): High risk adult recovery rate
        gcl (double): Low risk child recovery rate
        gch (double): High risk child recovery rate
        eps (double): External force of infection
        rhoal (double): Low risk adult susceptibility
        rhoah (double): High risk adult susceptibility
        rhocl (double): Low risk child susceptibility
        rhoch (double): High risk child susceptibility

    Returns:
        scipy.sparse.csr_matrix
    """

    cdef:
        #
        unsigned int sparsity_factor = 9

        # Calculate sub-block sizes
        int cSIZE = (Nc + 1) * (Nc + 2) // 2
        int aSIZE = (Na + 1) * (Na + 2) // 2
        # Calculate output matrix size
        int SIZE = cSIZE * aSIZE

        np.ndarray[np.double_t, ndim=1] D = np.zeros(sparsity_factor * SIZE)
        np.ndarray[np.int_t, ndim=1] r = np.zeros(sparsity_factor * SIZE, dtype=np.int)
        np.ndarray[np.int_t, ndim=1] c = np.zeros(sparsity_factor * SIZE, dtype=np.int)

        double Res, tmp;
        unsigned int sa, xl, xh, sc, yl, yh, I;
        int i = 0
        int i_tmp = 0

    # x{l, h} = Adults {l, h} Inf
    # y{l, h} = Children {l, h} Inf

    # Enumerate states for ADULT sub-blocks
    # (sa, xl, xh) = (Na, 0, 0)
    # (sa, xl, xh) = (Na - 1, 1, 0)
    # (sa, xl, xh) = (Na - 1, 0, 1)
    # (sa, xl, xh) = (Na - 2, 2, 0)
    #             ...
    # (sa, xl, xh) = (0, 1, Na - 1)
    # (sa, xl, xh) = (0, 0, Na)
    for sa in range(Na, -1, -1):
        for xl in range(Na - sa, -1, -1):
            xh = Na - xl - sa
            # Enumerate states for CHILD sub-blocks
            # (sc, yl, yh) = (Nc, 0, 0)
            # (sc, yl, yh) = (Nc - 1, 1, 0)
            # (sc, yl, yh) = (Nc - 1, 0, 1)
            # (sc, yl, yh) = (Nc - 2, 2, 0)
            #             ...
            # (sc, yl, yh) = (0, 1, Nc - 1)
            # (sc, yl, yh) = (0, 0, Nc)
            for sc in range(Nc, -1, -1):
                for yl in range(Nc - sc, -1, -1):
                    yh = Nc - yl - sc
                    # Index corresponding to current state
                    I = el_f(Na, Nc, xl, yl, sa, sc)

                    i_tmp = i

                    # Sum of force of infection
                    Res = bal * xl + bah * xh + bcl * yl + bch * yh + eps
                    # Calculate rate of flow to other states and get
                    # the corresponding indices
                    if sa >= 1:
                        c[i] = el_f(Na, Nc, xl + 1, yl, sa - 1, sc)
                        D[i] = Res * sa * rhoal  # Sa -> Ial
                        i += 1
                    if xl >= 1:
                        c[i] = el_f(Na, Nc, xl - 1, yl, sa + 1, sc)
                        D[i] = gal * xl  # Ial -> Sa
                        i += 1
                        c[i] = el_f(Na, Nc, xl - 1, yl, sa, sc)
                        D[i] = Res * xl * rhoah  # Ial -> Iah
                        i += 1
                    if xh >= 1:
                        c[i] = el_f(Na, Nc, xl + 1, yl, sa, sc)
                        D[i] = gah * xh  # Iah -> Ial
                        i += 1
                    if sc >= 1:
                        c[i] = el_f(Na, Nc, xl, yl + 1, sa, sc - 1)
                        D[i] = Res * sc * rhocl  # Sc -> Icl
                        i += 1
                    if yl >= 1:
                        c[i] = el_f(Na, Nc, xl, yl - 1, sa, sc + 1)
                        D[i] = gcl * yl  # Icl -> Sc
                        i += 1
                        c[i] = el_f(Na, Nc, xl, yl - 1, sa, sc)
                        D[i] = Res * yl * rhoch  # Icl -> Ich
                        i += 1
                    if yh >= 1:
                        c[i] = el_f(Na, Nc, xl, yl + 1, sa, sc)
                        D[i] = gch * yh  # Ich -> Icl
                        i += 1
                    r[i_tmp:i] = I

                    r[i] = I
                    c[i] = I
                    D[i] = -D[i_tmp:i].sum()
                    i += 1
    R = ssp.coo_matrix((D, (r, c)), shape=(SIZE, SIZE))
    R = R.tocsr()
    return R.T


cpdef SIS_A2R3_rho(unsigned int Na, unsigned int Nc, double bal, double bam, double bah, double bcl, double bcm, double bch, double gal, double gam, double gah, double gcl, double gcm, double gch, double eps, double rhoal, double rhoam, double rhoah, double rhocl, double rhocm, double rhoch, unsigned int cf):

    cdef int cSIZE = (Nc+1)*(Nc+2)*(Nc+3)/6
    cdef int aSIZE = (Na+1)*(Na+2)*(Na+3)/6
    cdef int SIZE = cSIZE*aSIZE
    #print("%d\t (%d,%d)"%(SIZE,Na,Nc))
    cdef np.ndarray[np.double_t, ndim=1] D = np.zeros(cf*SIZE)
    cdef np.ndarray[np.int_t, ndim=1] r = np.zeros(cf*SIZE,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] c = np.zeros(cf*SIZE,dtype=np.int)
    #x{l,h} = Adults {l,h} Inf
    #y{l,h} = Children {l,h} Inf

    cdef double Res, tmp;
    cdef unsigned int sa, xl, xm, xh, sc, yl, ym, yh, I;
    cdef int i = 0
    for sa in range(Na,-1,-1):
        for xl in range(Na-sa,-1,-1): #Decreasing low risk each sc loop
            for xm in range(Na-sa-xl,-1,-1): #Decreasing low risk each sc loop
                xh = Na-xl-sa-xm #Increasing high risk each sc loop
                assert sa+xl+xm+xh == Na, (Na, sa, xl, xm, xh)
                #CHILD SUB-BLOCK
                for sc in range(Nc,-1,-1): #Decreasing susceptibles
                    for yl in range(Nc-sc,-1,-1): #Decreasing low risk each sc loop
                        for ym in range(Nc-sc-yl,-1,-1): #Decreasing low risk each sc loop
                            yh = Nc-yl-sc-ym #Increasing high risk each sc loop
                            assert sc+yl+ym+yh == Nc, (Nc, sc, yl, ym, yh)
                            I = el_f_A2_R3(Na,Nc,xl,xm,yl,ym,sa,sc)
                            #print(I,"\t",sc,yl,yh)

                            tmp=0
                            Res = bal*xl+bam*xm+bah*xh+bcl*yl+bcm*ym+bch*yh+eps #Sum of FOI
                            if sa>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl+1,xm,yl,ym,sa-1,sc)
                                D[i]=Res*sa*rhoal  # Sa->Iam
                                i+=1
                                tmp+=Res*sa*rhoal
                            if xl>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl-1,xm,yl,ym,sa+1,sc)
                                D[i]=gal*xl# Ial->Sa
                                i+=1
                                tmp+=gal*xl
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl-1,xm+1,yl,ym,sa,sc)
                                D[i]=Res*xl*rhoam  # Ial->Iam
                                i+=1
                                tmp+=Res*xl*rhoam
                            if xm>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl+1,xm-1,yl,ym,sa,sc)
                                D[i]=gam*xm# Iam->Ial
                                i+=1
                                tmp+=gam*xm
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm-1,yl,ym,sa,sc)
                                D[i]=Res*xm*rhoah  # Iam->Iah
                                i+=1
                                tmp+=Res*xm*rhoah
                            if xh>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm+1,yl,ym,sa,sc)
                                D[i]=gah*xh  # Iah->Iam
                                i+=1
                                tmp+=gah*xh
                            if sc>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl+1,ym,sa,sc-1)
                                D[i]=Res*sc*rhocl  # Sc->Icl
                                i+=1
                                tmp+=Res*sc*rhocl
                            if yl>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl-1,ym,sa,sc+1)
                                D[i]=gcl*yl# Icl->Sc
                                i+=1
                                tmp+=gcl*yl
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl-1,ym+1,sa,sc)
                                D[i]=Res*yl*rhocm  # Icl->Icm
                                i+=1
                                tmp+=Res*yl*rhocm
                            if ym>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl+1,ym-1,sa,sc)
                                D[i]=gcm*ym# Icm->Icl
                                i+=1
                                tmp+=gcm*ym
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl,ym-1,sa,sc)
                                D[i]=Res*ym*rhoch  # Icm->Ich
                                i+=1
                                tmp+=Res*ym*rhoch
                            if yh>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl,ym+1,sa,sc)
                                D[i]=gch*yh# Ich->Icm
                                i+=1
                                tmp+=gch*yh
                            r[i]=I
                            c[i]=I
                            D[i]=-tmp
                            i+=1
                            #print(i,7*SIZE)
    R= ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE))
    R=R.tocsr()
    return R.T
