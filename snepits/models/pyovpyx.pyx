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


@lru_cache()
def el_f_gen_memo(np.ndarray[np.int_t, ndim=1] N, np.ndarray[np.int_t, ndim=1] I):
    return el_f_gen_base(N, I)

cpdef unsigned int el_f_gen(np.ndarray[np.int_t, ndim=1] N, np.ndarray[np.int_t, ndim=1] I):
    return el_f_gen_base(N, I)

cdef unsigned int el_f_gen_base(np.ndarray[np.int_t, ndim=1] N, np.ndarray[np.int_t, ndim=1] I):
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

cpdef unsigned int el_f(unsigned int Na,unsigned int Nc,unsigned int xl,unsigned int yl,unsigned int sa,unsigned int sc):
    return (Nc+1)*(Nc+2)/2 * ( (Na-sa-xl)+<unsigned int>((Na-sa)*(Na-sa+1)*0.5)) + (Nc-sc-yl) + \
            <unsigned int>((Nc-sc)*(Nc-sc+1)*0.5 )

cpdef unsigned int el_f_A2_R3(unsigned int Na, unsigned int Nc, unsigned int xl, unsigned int xm, unsigned int yl, unsigned int ym, unsigned int sa, unsigned int sc):
    return <unsigned int>  (Nc+1)*(Nc+2)*(Nc+3)/6 * ( (Na-sa-xl-xm)+((Na-sa-xl)*(Na-sa-xl+1)/2)+((Na-sa)*(Na-sa+1)*(Na-sa+2)/6) ) + ( (Nc-sc-yl-ym)+((Nc-sc-yl)*(Nc-sc-yl+1)/2)+((Nc-sc)*(Nc-sc+1)*(Nc-sc+2)/6) )

cpdef unsigned int el_f_A3_R3(unsigned int Na, unsigned int Nc, unsigned int Ni, unsigned int xl, unsigned int xm, unsigned int yl, unsigned int ym, unsigned int zl, unsigned int zm, unsigned int sa, unsigned int sc, unsigned int si):
    return <unsigned int>  (Ni+1)*(Ni+2)*(Ni+3)/6 * (Nc+1)*(Nc+2)*(Nc+3)/6 * ( (Na-sa-xl-xm)+((Na-sa-xl)*(Na-sa-xl+1)/2)+((Na-sa)*(Na-sa+1)*(Na-sa+2)/6) ) +\
            (Ni+1)*(Ni+2)*(Ni+3)/6*( (Nc-sc-yl-ym)+((Nc-sc-yl)*(Nc-sc-yl+1)/2)+((Nc-sc)*(Nc-sc+1)*(Nc-sc+2)/6) ) +\
            ( (Ni-si-zl-zm)+((Ni-si-zl)*(Ni-si-zl+1)/2)+((Ni-si)*(Ni-si+1)*(Ni-si+2)/6) )

cpdef SIS_A2R3_sp_arr(unsigned int Na, unsigned int Nc, double bal, double bam, double bah, double bcl, double bcm, double bch,\
               double gal, double gam, double gah, double gcl, double gcm, double gch, double eps,double rhoa,double rhoc, unsigned int cf):

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
                                D[i]=Res*sa*rhoa# Sa->Iam
                                i+=1
                                tmp+=Res*sa*rhoa
                            if xl>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl-1,xm,yl,ym,sa+1,sc)
                                D[i]=gal*xl# Ial->Sa
                                i+=1
                                tmp+=gal*xl
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl-1,xm+1,yl,ym,sa,sc)
                                D[i]=Res*xl*rhoa# Ial->Iam
                                i+=1
                                tmp+=Res*xl*rhoa
                            if xm>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl+1,xm-1,yl,ym,sa,sc)
                                D[i]=gam*xm# Iam->Ial
                                i+=1
                                tmp+=gam*xm
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm-1,yl,ym,sa,sc)
                                D[i]=Res*xm*rhoa# Iam->Iah
                                i+=1
                                tmp+=Res*xm*rhoa
                            if xh>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm+1,yl,ym,sa,sc)
                                D[i]=gah*xh# Iah->Iam
                                i+=1
                                tmp+=gah*xh
                            if sc>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl+1,ym,sa,sc-1)
                                D[i]=Res*sc*rhoc# Sc->Icl
                                i+=1
                                tmp+=Res*sc*rhoc
                            if yl>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl-1,ym,sa,sc+1)
                                D[i]=gcl*yl# Icl->Sc
                                i+=1
                                tmp+=gcl*yl
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl-1,ym+1,sa,sc)
                                D[i]=Res*yl*rhoc# Icl->Icm
                                i+=1
                                tmp+=Res*yl*rhoc
                            if ym>=1:
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl+1,ym-1,sa,sc)
                                D[i]=gcm*ym# Icm->Icl
                                i+=1
                                tmp+=gcm*ym
                                r[i]=I
                                c[i]=el_f_A2_R3(Na,Nc,xl,xm,yl,ym-1,sa,sc)
                                D[i]=Res*ym*rhoc# Icm->Ich
                                i+=1
                                tmp+=Res*ym*rhoc
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

cpdef SIS_A3R3_rho(unsigned int Na, unsigned int Nc, unsigned int Ni, double bal, double bam, double bah, double bcl, double bcm, double bch, double bil, double bim, double bih, double gal, double gam, double gah, double gcl, double gcm, double gch, double gil, double gim, double gih, double eps, double rhoal, double rhoam, double rhoah, double rhocl, double rhocm, double rhoch, double rhoil, double rhoim, double rhoih, unsigned int cf):

    cdef int cSIZE = (Nc+1)*(Nc+2)*(Nc+3)/6
    cdef int aSIZE = (Na+1)*(Na+2)*(Na+3)/6
    cdef int iSIZE = (Ni+1)*(Ni+2)*(Ni+3)/6
    cdef int SIZE = iSIZE*cSIZE*aSIZE
    #print("%d\t (%d,%d)"%(SIZE,Na,Nc))
    cdef np.ndarray[np.double_t, ndim=1] D = np.zeros(cf*SIZE)
    cdef np.ndarray[np.int_t, ndim=1] r = np.zeros(cf*SIZE,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] c = np.zeros(cf*SIZE,dtype=np.int)
    #x{l,h} = Adults {l,h} Inf
    #y{l,h} = Children {l,h} Inf
    #z{l,h} = Infants (pre-SAC) {l,h} Inf

    cdef double Res, tmp;
    cdef unsigned int sa, xl, xm, xh, sc, yl, ym, yh, zl, zm, zh, I;
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

                            for si in range(Ni, -1, -1):
                                for zl in range(Ni-si, -1, -1):
                                    for zm in range(Ni-si-zl, -1, -1):
                                        zh = Ni-si-zl-zm
                                        assert si+zl+zm+zh == Ni, (Ni, si, zl, zm, zh)
                                        I = el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl,zm,sa,sc,si)
                                        #print(I,"\t",sc,yl,yh)

                                        tmp=0
                                        Res = bal*xl+bam*xm+bah*xh+bcl*yl+bcm*ym+bch*yh+bil*zl+bim*zm+bih*zh+eps #Sum of FOI
                                        if sa>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl+1,xm,yl,ym,zl,zm,sa-1,sc,si)
                                            D[i]=Res*sa*rhoal  # Sa->Iam
                                            i+=1
                                            tmp+=Res*sa*rhoal
                                        if xl>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl-1,xm,yl,ym,zl,zm,sa+1,sc,si)
                                            D[i]=gal*xl# Ial->Sa
                                            i+=1
                                            tmp+=gal*xl
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl-1,xm+1,yl,ym,zl,zm,sa,sc,si)
                                            D[i]=Res*xl*rhoam  # Ial->Iam
                                            i+=1
                                            tmp+=Res*xl*rhoam
                                        if xm>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl+1,xm-1,yl,ym,zl,zm,sa,sc,si)
                                            D[i]=gam*xm# Iam->Ial
                                            i+=1
                                            tmp+=gam*xm
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm-1,yl,ym,zl,zm,sa,sc,si)
                                            D[i]=Res*xm*rhoah  # Iam->Iah
                                            i+=1
                                            tmp+=Res*xm*rhoah
                                        if xh>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm+1,yl,ym,zl,zm,sa,sc,si)
                                            D[i]=gah*xh  # Iah->Iam
                                            i+=1
                                            tmp+=gah*xh
                                        if sc>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl+1,ym,zl,zm,sa,sc-1,si)
                                            D[i]=Res*sc*rhocl  # Sc->Icl
                                            i+=1
                                            tmp+=Res*sc*rhocl
                                        if yl>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl-1,ym,zl,zm,sa,sc+1,si)
                                            D[i]=gcl*yl# Icl->Sc
                                            i+=1
                                            tmp+=gcl*yl
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl-1,ym+1,zl,zm,sa,sc,si)
                                            D[i]=Res*yl*rhocm  # Icl->Icm
                                            i+=1
                                            tmp+=Res*yl*rhocm
                                        if ym>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl+1,ym-1,zl,zm,sa,sc,si)
                                            D[i]=gcm*ym# Icm->Icl
                                            i+=1
                                            tmp+=gcm*ym
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym-1,zl,zm,sa,sc,si)
                                            D[i]=Res*ym*rhoch  # Icm->Ich
                                            i+=1
                                            tmp+=Res*ym*rhoch
                                        if yh>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym+1,zl,zm,sa,sc,si)
                                            D[i]=gch*yh# Ich->Icm
                                            i+=1
                                            tmp+=gch*yh
                                        if si>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl+1,zm,sa,sc,si-1)
                                            D[i]=Res*si*rhoil  # Si->Iil
                                            i+=1
                                            tmp+=Res*si*rhoil
                                        if zl>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl-1,zm,sa,sc,si+1)
                                            D[i]=gil*zl# Iil->Si
                                            i+=1
                                            tmp+=gil*zl
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl-1,zm+1,sa,sc,si)
                                            D[i]=Res*zl*rhoim  # Iil->Iim
                                            i+=1
                                            tmp+=Res*zl*rhoim
                                        if zm>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl+1,zm-1,sa,sc,si)
                                            D[i]=gim*zm# Iim->Iil
                                            i+=1
                                            tmp+=gim*zm
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl,zm-1,sa,sc,si)
                                            D[i]=Res*zm*rhoih  # Iim->Iih
                                            i+=1
                                            tmp+=Res*zm*rhoih
                                        if zh>=1:
                                            r[i]=I
                                            c[i]=el_f_A3_R3(Na,Nc,Ni,xl,xm,yl,ym,zl,zm+1,sa,sc,si)
                                            D[i]=gih*zh# Iih->Iim
                                            i+=1
                                            tmp+=gih*zh
                                        r[i]=I
                                        c[i]=I
                                        D[i]=-tmp
                                        i+=1
                                        #print(i,7*SIZE)
    R= ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE))
    R=R.tocsr()
    return R.T

cpdef SIS_ACR_sp_arr(unsigned int Na, unsigned int Nc, double bal, double bah, double bcl, double bch,\
               double gal, double gah, double gcl, double gch, double eps,double rhoa,double rhoc):

    cdef int cSIZE = (Nc+1)*(Nc+2)/2
    cdef int aSIZE = (Na+1)*(Na+2)/2
    cdef int SIZE = cSIZE*aSIZE
    #print("%d\t (%d,%d)"%(SIZE,Na,Nc))
    cdef np.ndarray[np.double_t, ndim=1] D = np.zeros(9*SIZE)
    cdef np.ndarray[np.int_t, ndim=1] r = np.zeros(9*SIZE,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] c = np.zeros(9*SIZE,dtype=np.int)
    #x{l,h} = Adults {l,h} Inf
    #y{l,h} = Children {l,h} Inf

    cdef double Res, tmp;
    cdef unsigned int sa, xl, xh, sc, yl, yh, I;
    cdef int i = 0
    for sa in range(Na,-1,-1):
        for xl in range(Na-sa,-1,-1): #Decreasing low risk each sc loop
            xh = Na-xl-sa #Increasing high risk each sc loop
            #CHILD SUB-BLOCK
            for sc in range(Nc,-1,-1): #Decreasing susceptibles
                for yl in range(Nc-sc,-1,-1): #Decreasing low risk each sc loop
                    yh = Nc-yl-sc #Increasing high risk each sc loop
                    I = el_f(Na,Nc,xl,yl,sa,sc)
                    #print(I,"\t",sc,yl,yh)

                    tmp=0
                    Res = bal*xl+bah*xh+bcl*yl+bch*yh+eps #Sum of FOI
                    if sa>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl+1,yl,sa-1,sc)
                        D[i]=Res*sa*rhoa# Sa->Ial
                        i+=1
                        tmp+=Res*sa*rhoa
                    if xl>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl-1,yl,sa+1,sc)
                        D[i]=gal*xl# Ial->Sa
                        i+=1
                        tmp+=gal*xl
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl-1,yl,sa,sc)
                        D[i]=Res*xl*rhoa# Ial->Iah
                        i+=1
                        tmp+=Res*xl*rhoa
                    if xh>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl+1,yl,sa,sc)
                        D[i]=gah*xh# Iah->Ial
                        i+=1
                        tmp+=gah*xh
                    if sc>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl+1,sa,sc-1)
                        D[i]=Res*sc*rhoc# Sc->Icl
                        i+=1
                        tmp+=Res*sc*rhoc
                    if yl>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl-1,sa,sc+1)
                        D[i]=gcl*yl# Icl->Sc
                        i+=1
                        tmp+=gcl*yl
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl-1,sa,sc)
                        D[i]=Res*yl*rhoc# Icl->Ich
                        i+=1
                        tmp+=Res*yl*rhoc
                    if yh>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl+1,sa,sc)
                        #assert(el_f(Na,Nc,xl,yl+1,sa,sc)==el_f2(Na,Nc,xl,yl+1,sa,sc))
                        D[i]=gch*yh# Ich->Icl
                        i+=1
                        tmp+=gch*yh
                    r[i]=I
                    c[i]=I
                    D[i]=-tmp
                    i+=1
                    #print(i,7*SIZE)
    #R= ssp.coo_matrix((D[0:i],(r[0:i],c[0:i])),shape=(SIZE,SIZE))
    R= ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE))
    R=R.tocsr()
    return R.T

cpdef SIS_ACR_sp_arr2(unsigned int Na, unsigned int Nc, double bal, double bah, double bcl, double bch,\
               double gal, double gah, double gcl, double gch, double eps, double rhoal, double rhoah, double rhocl, double rhoch):

    cdef int cSIZE = (Nc+1)*(Nc+2)/2
    cdef int aSIZE = (Na+1)*(Na+2)/2
    cdef int SIZE = cSIZE*aSIZE
    #print("%d\t (%d,%d)"%(SIZE,Na,Nc))
    cdef np.ndarray[np.double_t, ndim=1] D = np.zeros(9*SIZE)
    cdef np.ndarray[np.int_t, ndim=1] r = np.zeros(9*SIZE,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] c = np.zeros(9*SIZE,dtype=np.int)
    #x{l,h} = Adults {l,h} Inf
    #y{l,h} = Children {l,h} Inf

    cdef double Res,tmp;
    cdef unsigned int sa,xl,xh,sc,yl,yh,I;
    cdef int i=0
    for sa in range(Na,-1,-1):
        for xl in range(Na-sa,-1,-1): #Decreasing low risk each sc loop
            xh = Na-xl-sa #Increasing high risk each sc loop
            #CHILD SUB-BLOCK
            for sc in range(Nc,-1,-1): #Decreasing susceptibles
                for yl in range(Nc-sc,-1,-1): #Decreasing low risk each sc loop
                    yh = Nc-yl-sc #Increasing high risk each sc loop
                    I = el_f(Na,Nc,xl,yl,sa,sc)
                    #print(I,"\t",sc,yl,yh)

                    tmp=0
                    Res = bal*xl+bah*xh+bcl*yl+bch*yh+eps #Sum of FOI
                    if sa>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl+1,yl,sa-1,sc)
                        D[i]=Res*sa*rhoal# Sa->Ial
                        i+=1
                        tmp+=Res*sa*rhoal
                    if xl>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl-1,yl,sa+1,sc)
                        D[i]=gal*xl# Ial->Sa
                        i+=1
                        tmp+=gal*xl
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl-1,yl,sa,sc)
                        D[i]=Res*xl*rhoah# Ial->Iah
                        i+=1
                        tmp+=Res*xl*rhoah
                    if xh>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl+1,yl,sa,sc)
                        D[i]=gah*xh# Iah->Ial
                        i+=1
                        tmp+=gah*xh
                    if sc>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl+1,sa,sc-1)
                        D[i]=Res*sc*rhocl# Sc->Icl
                        i+=1
                        tmp+=Res*sc*rhocl
                    if yl>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl-1,sa,sc+1)
                        D[i]=gcl*yl# Icl->Sc
                        i+=1
                        tmp+=gcl*yl
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl-1,sa,sc)
                        D[i]=Res*yl*rhoch# Icl->Ich
                        i+=1
                        tmp+=Res*yl*rhoch
                    if yh>=1:
                        r[i]=I
                        c[i]=el_f(Na,Nc,xl,yl+1,sa,sc)
                        #assert(el_f(Na,Nc,xl,yl+1,sa,sc)==el_f2(Na,Nc,xl,yl+1,sa,sc))
                        D[i]=gch*yh# Ich->Icl
                        i+=1
                        tmp+=gch*yh
                    r[i]=I
                    c[i]=I
                    D[i]=-tmp
                    i+=1
                    #print(i,7*SIZE)
    #R= ssp.coo_matrix((D[0:i],(r[0:i],c[0:i])),shape=(SIZE,SIZE))
    R= ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE))
    R=R.tocsr()
    return R.T

cpdef unsigned int el_f_R3(unsigned int N, unsigned int xl, unsigned int xm,
                           unsigned int xh, unsigned int s):
    return (N-s)*(N-s+1)*(N-s+2)/6 + (xm+xh)*(xm+xh+1)/2 + xh

cpdef SIS_R3_mat(unsigned int N, double bl, double bm, double bh,
             double gl, double gm, double gh, double eps,
             double rhol, double rhom, double rhoh):

    cdef int SIZE = (N+1)*(N+2)*(N+3)/6
    #print("%d\t (%d,%d)"%(SIZE,Na,Nc))
    cdef np.ndarray[np.double_t, ndim=1] D = np.zeros(9*SIZE)
    cdef np.ndarray[np.int_t, ndim=1] r = np.zeros(9*SIZE,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] c = np.zeros(9*SIZE,dtype=np.int)
    #x{l,m,h} = {l,m,h} Inf

    cdef double Res,tmp;
    cdef unsigned int s,xl,xm,xh,I;
    cdef int i=0
    # N = s + xl + xm + xh
    for s in range(N,-1,-1):
        for xl in range(N-s,-1,-1): #Decreasing low risk each sc loop
            for xm in range(N - s - xl,-1,-1): #Decreasing medium risk each sc loop
                xh = N - s - xl - xm
                I = el_f_R3(N,xl,xm,xh,s)

                tmp=0
                Res = bl*xl+bm*xm+bh*xh+eps #Sum of FOI
                if s>=1:
                    r[i]=I
                    c[i]=el_f_R3(N,xl+1,xm,xh,s-1)
                    D[i]=Res*s*rhol  # S->Il
                    i+=1
                    tmp+=Res*s*rhol
                if xl>=1:
                    r[i]=I
                    c[i]=el_f_R3(N,xl-1,xm,xh,s+1)
                    D[i]=gl*xl  # Il->S
                    i+=1
                    tmp+=gl*xl
                    r[i]=I
                    c[i]=el_f_R3(N,xl-1,xm+1,xh,s)
                    D[i]=Res*xl*rhom# Il->Im
                    i+=1
                    tmp+=Res*xl*rhom
                if xm>=1:
                    r[i]=I
                    c[i]=el_f_R3(N,xl+1,xm-1,xh,s)
                    D[i]=gm*xm# Im->Il
                    i+=1
                    tmp+=gm*xm
                    r[i]=I
                    c[i]=el_f_R3(N,xl,xm-1,xh+1,s)
                    D[i]=Res*xm*rhoh# Im->Ih
                    i+=1
                    tmp+=Res*xm*rhoh
                if xh>=1:
                    r[i]=I
                    c[i]=el_f_R3(N,xl,xm+1,xh-1,s)
                    D[i]=gh*xh# Ih->Im
                    i+=1
                    tmp+=gh*xh
                r[i]=I
                c[i]=I
                D[i]=-tmp
                i+=1
                #print(i,7*SIZE)
    #R= ssp.coo_matrix((D[0:i],(r[0:i],c[0:i])),shape=(SIZE,SIZE))
    R= ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE))
    R=R.tocsr()
    return R.T

cpdef unsigned int el_f_AC_R3(unsigned int Na, unsigned int Nc,
                        unsigned int xl, unsigned int xm, unsigned int xh,
                        unsigned int yl, unsigned int ym, unsigned int yh,
                        unsigned int sa, unsigned int sc):
    return (Nc+1)*(Nc+2)*(Nc+3)/6 * ((Na-sa)*(Na-sa+1)*(Na-sa+2)/6 +\
                                     (xm+xh)*(xm+xh+1)/2 + xh) + \
        (Nc-sc)*(Nc-sc+1)*(Nc-sc+2)/6 + (ym+yh)*(ym+yh+1)/2 + yh

cpdef SIS_AC_R3_mat(unsigned int Na, unsigned int Nc, double bal, double bam, double bah,
                    double bcl, double bcm, double bch,
                    double gl, double gm, double gh, double eps,
                    double rhoAl, double rhoAm, double rhoAh,
                    double rhoCl, double rhoCm, double rhoCh):

    cdef int cSIZE = (Nc+1)*(Nc+2)*(Nc+3)/6
    cdef int aSIZE = (Na+1)*(Na+2)*(Na+3)/6
    cdef int SIZE = cSIZE*aSIZE

    cdef unsigned int comp_factor = 100
    #print("%d\t (%d,%d)"%(SIZE,Na,Nc))
    cdef np.ndarray[np.double_t, ndim=1] D = np.zeros(comp_factor*SIZE)
    cdef np.ndarray[np.int_t, ndim=1] r = np.zeros(comp_factor*SIZE,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] c = np.zeros(comp_factor*SIZE,dtype=np.int)
    #x{l,m,h} = Adults {l,m,h} Inf
    #y{l,m,h} = Child {l,m,h} Inf

    cdef double Res, tmp;
    cdef unsigned int s, xl, xm, xh, yl, ym, yh, I;
    cdef int i = 0
    # Na = sa + xl + xm + xh
    # Nc = sc + yl + ym + yh
    for sa in range(Na,-1,-1):
        for xl in range(Na-sa,-1,-1): #Decreasing low risk each sc loop
            for xm in range(Na - sa - xl,-1,-1): #Decreasing medium risk each sc loop
                xh = Na - sa - xl - xm

                # CHILD SUBBLOCK
                for sc in range(Nc,-1,-1):
                    for yl in range(Nc-sc,-1,-1): #Decreasing low risk each sc loop
                        for ym in range(Nc - sc - yl,-1,-1): #Decreasing medium risk each sc loop
                            yh = Nc - sc - yl - ym

                            I = el_f_AC_R3(Na, Nc, xl, xm, xh, yl, ym, yh, sa, sc)

                            tmp=0
                            Res = bal*xl+bam*xm+bah*xh +bcl*yl+bcm*ym+bch*yh +eps #Sum of FOI
                            if sa>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl+1, xm, xh, yl, ym, yh, sa-1, sc)
                                D[i]=Res*sa*rhoAl  # Sa->Ial
                                i+=1
                                tmp+=Res*sa*rhoAl
                            if xl>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl-1, xm, xh, yl, ym, yh, sa+1, sc)
                                D[i]=gl*xl  # Ial->S
                                i+=1
                                tmp+=gl*xl
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl-1, xm+1, xh, yl, ym, yh, sa, sc)
                                D[i]=Res*xl*rhoAm  # Ial->Iam
                                i+=1
                                tmp+=Res*xl*rhoAm
                            if xm>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl+1, xm-1, xh, yl, ym, yh, sa, sc)
                                D[i]=gm*xm  # Iam->Ial
                                i+=1
                                tmp+=gm*xm
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm-1, xh+1, yl, ym, yh, sa, sc)
                                D[i]=Res*xm*rhoAh  # Iam->Iah
                                i+=1
                                tmp+=Res*xm*rhoAh
                            if xh>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm+1, xh-1, yl, ym, yh, sa, sc)
                                D[i]=gh*xh  # Iah->Iam
                                i+=1
                                tmp+=gh*xh
                            if sc>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm, xh, yl+1, ym, yh, sa, sc-1)
                                D[i]=Res*sc*rhoCl  # Sc->Icl
                                i+=1
                                tmp+=Res*sc*rhoCl
                            if yl>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm, xh, yl-1, ym, yh, sa, sc+1)
                                D[i]=gl*yl  # Icl->S
                                i+=1
                                tmp+=gl*yl
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm, xh, yl-1, ym+1, yh, sa, sc)
                                D[i]=Res*yl*rhoCm  # Icl->Icm
                                i+=1
                                tmp+=Res*yl*rhoCm
                            if ym>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm, xh, yl+1, ym-1, yh, sa, sc)
                                D[i]=gm*ym  # Icm->Icl
                                i+=1
                                tmp+=gm*ym
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm, xh, yl, ym-1, yh+1, sa, sc)
                                D[i]=Res*ym*rhoCh  # Icm->Ich
                                i+=1
                                tmp+=Res*ym*rhoCh
                            if yh>=1:
                                r[i]=I
                                c[i]=el_f_AC_R3(Na, Nc, xl, xm, xh, yl, ym+1, yh-1, sa, sc)
                                D[i]=gh*yh  # Ich->Icm
                                i+=1
                                tmp+=gh*yh
                            r[i]=I
                            c[i]=I
                            D[i]=-tmp
                            i+=1
                            #print(i,7*SIZE)
    #R= ssp.coo_matrix((D[0:i],(r[0:i],c[0:i])),shape=(SIZE,SIZE))
    R= ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE))
    R=R.tocsr()
    return R.T


cdef np.ndarray[np.int_t, ndim=1] f(unsigned int R, unsigned int A, unsigned int i, unsigned int a):
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
    # X{r, a} = risk level r & age class a
    cdef:
        unsigned int SIZE = barsize(A, R, Ntup)
        unsigned int N = Ntup.sum()
        np.ndarray[np.double_t, ndim=1] D = np.zeros(cf*SIZE, dtype=np.double)
        np.ndarray[np.int_t, ndim=1] r = np.zeros(cf*SIZE, dtype=np.int)
        np.ndarray[np.int_t, ndim=1] c = np.zeros(cf*SIZE, dtype=np.int)
    beta = beta * 1/(<double> N-1)**alpha

    cdef unsigned int i = 0
    cdef np.ndarray[np.int_t, ndim=2] X = np.empty((R+1, A), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] Xf = np.empty(R+1*A, dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] Xi = np.empty(R+1, dtype=np.int)
    cdef unsigned int I
    cdef double Res, tmp
    cdef size_t a, ri
    cdef unsigned int Xij

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

    return ssp.coo_matrix((D,(r,c)),shape=(SIZE,SIZE)).tocsr().T

cpdef treat_rand_mat_pyx(double coverage, double efficacy, np.ndarray[np.int_t, ndim=1] Nsubset, np.ndarray[np.int_t, ndim=1] Ntup, int A, int R, int Msize):
    from pyx_funs import idx_to_state, el_f_gen
    from numpy import zeros, product, diag, empty, arange
    from scipy.stats import binom
    from itertools import product as itproduct
    from itertools import chain

    cdef np.ndarray[np.double_t, ndim=2] binom_precalc = zeros((Ntup.max()+1, Ntup.max()+1))
    cdef:
        size_t n, k, stdix, i
        unsigned int stidx2
        double Psel
    for n in range(Ntup.max()+1):
        for k in range(n+1):
            binom_precalc[k, n] = binom.pmf(k, n=n, p=coverage*efficacy)

    cdef np.ndarray[np.double_t, ndim=2] QP = zeros((Msize, Msize))  # TODO SPARSE
    cdef np.ndarray[np.int_t, ndim=1] sub_idx = empty(Nsubset.shape[0]*R, dtype=np.int)
    i = 0
    for n in range(Nsubset.shape[0]):
        sub_idx[i:i+R] = arange(Nsubset[n]*R, (Nsubset[n]+1)*R)
        i += R

    cdef np.ndarray[np.int_t, ndim=1] S = empty(A*R, dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] Sprime = empty(A*R, dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] Sprime_l
    for stidx in range(Msize):
        S = idx_to_state(stidx, A, R, Ntup)  # get state representation
        Sprime_l = get_Sprime_l(S, R, Nsubset)  # calculate modified states
        for Sprime in Sprime_l:
            Psel = 1
            for i in range(len(sub_idx)):
                Psel *= binom_precalc[(S-Sprime)[sub_idx[i]], S[sub_idx[i]]]
            stidx2 = el_f_gen(Ntup, Sprime)
            QP[stidx2, stidx] = Psel

    QP = (QP + diag(1-QP.sum(axis=0)))  # Assign probabilites of staying in state S
    return QP


cdef np.ndarray[np.int_t, ndim=2] get_Sprime_l(np.ndarray[np.int_t, ndim=1] S, int R, np.ndarray[np.int_t, ndim=1] Nsubset):
    from numpy import empty, indices

    cdef size_t i
    cdef np.ndarray[np.int_t, ndim=1] ip = S+1  # empty(S.shape[0], dtype=np.int)
    for i in range(S.shape[0]):
        if i//R not in Nsubset:
            ip[i] = 1

    cdef np.ndarray[np.int_t, ndim=2] cartesian_ip = indices(ip).T.reshape((-1, S.shape[0]))
    cdef np.ndarray[np.int_t, ndim=2] Sprime_l = empty((cartesian_ip.size//S.shape[0], S.shape[0]), dtype=np.int)
    for i in range(cartesian_ip.shape[0]):
        Sprime_l[i, :] = S-cartesian_ip[i]
    return Sprime_l
