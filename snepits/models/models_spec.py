"""
from models_spec import SIS, SIS_pop

# Single household
model = SIS(10, (2, 0.5), freq_dep=True)

# Multiple households - no data
model = SIS_pop([10, 10], (2, 0.5), freq_dep=True, data=None)

# Multiple households - data
model = SIS_pop([10, 10], (2, 0.5), freq_dep=True, data=[4, 5])
"""
import logging

from numpy import arange, diag

import snepits
from snepits.models.models_meta import Model, Population

logger = logging.getLogger(__name__)


class SIS(Model):
    """ SIS model class.

    Attributes:
        param_l (list[str]): Name of parameters
        risk_l (list[str]): Name of risk levels
        dim (int): Number of parameters
        sparse (bool): If True use sparse solver


    Args:
        N (int): Population size
        params (tuple): Model parameters (beta, eps)
            beta - Within population transmission rate
            eps - External Force of infection
        freq_dep (bool, optional): Frequency dependent transmission if True
    """

    param_l = ['beta', 'eps', 'alpha']
    risk_l = ['N']
    R = 1
    dim = len(param_l)
    sparse = False

    def gen_mat(self):
        """
        Generate transition matrix `M` of model
        """
        N = self.N
        beta = self.beta
        alpha = self.alpha
        if self.freq_dep:
            beta *= 1 / (N - 1)**alpha
        eps = self.eps

        x = arange(N + 1)
        self.M = diag(x[1:], k=1) +\
            diag((N - x[:-1]) * (beta * x[:-1] + eps), k=-1)
        self.M[x, x] = - self.M.sum(axis=0)
        return self.M

    def __str__(self):
        return 'SIS: N = %d' % (self.N)


class SIS_pop(Population):
    """ SIS_pop model class.

    Attributes:
        subclass (obj): Class of sub_pops
        param_l (list[str]): Name of parameters
        risk_l (list[str]): Name of risk classes
        dim (int): Number of parameters

    Args:
        sizes (numpy.array): Array of population sizes for each class.
            One row per household, one column per class.
        params (tuple): Model parameters (beta, eps)
            beta - Within population transmission rate
            eps - External Force of infection
        freq_dep (bool, optional): Frequency dependent transmission if True
        data (numpy.array, optional): Array of numbers infected for each class
            One row per household, one column per class.
    """

    subclass = SIS
    param_l = subclass.param_l
    risk_l = subclass.risk_l
    dim = len(param_l)
    R = subclass.R

    def data_transform(self, data):
        return data

    def __str__(self):
        return 'SIS_pop: m = %d, n = %d' % (self.m, self.N)


class SIS_AC(Model):
    '''
    Initialise SIS_AC model class.
    Inputs:
    Na - Adult population size
    Nc - Child population size
    params (tuple)  beta_A - Adult within population transmission rate
                    beta_C - Child within population transmission rate
                    rho_C  - Child susceptibility
                    eps - External FOI
    freq_dep (bool, optional) - frequency dependence
    NOTE: rho_A  - Adult susceptibility - is always 1
    '''
    param_l = ['beta_A', 'beta_C', 'rho_C', 'eps', 'alpha']
    risk_l = ['Na', 'Nc']
    R = 1
    dim = len(param_l)
    sparse = False

    def __str__(self):
        return 'SIS_AC: N = %d, Na = %d, Nc = %d' % (self.N, self.Na, self.Nc)

    def gen_mat(self):
        '''
        Na, Nc, bal, bah, bcl, bch, gl, ra(=1),  rc, gl, eps
        '''
        from scipy.sparse import coo_matrix
        import numpy as np
        N = self.N
        Na = self.Na
        Nc = self.Nc
        ba = self.beta_A
        bc = self.beta_C
        alpha = self.alpha

        rhoa = 1
        rhoc = self.rho_C
        if self.freq_dep:
            ba *= 1 / (N - 1) ** alpha
            bc *= 1 / (N - 1) ** alpha
        eps = self.eps
        g = 1

        SIZE = (Na+1)*(Nc+1)
        D = np.zeros(9 * SIZE)
        r = np.zeros(9 * SIZE, dtype=np.int)
        c = np.zeros(9 * SIZE, dtype=np.int)
        # x{l,h} = Adults {l,h} Inf
        # y{l,h} = Children {l,h} Inf

        def el_f(Na, Nc, xl, yl, sa, sc):
            return xl*(Nc+1) + yl

        i = 0
        for sa in range(Na, -1, -1):
            xl = Na - sa  # Decreasing low risk each sc loop
            # CHILD SUB-BLOCK
            for sc in range(Nc, -1, -1):  # Decreasing susceptibles
                yl = Nc - sc  # Decreasing low risk each sc loop
                I = el_f(Na, Nc, xl, yl, sa, sc)

                tmp = 0
                Res = ba * xl + bc * yl + eps  # Sum of FOI
                # print(I,'\t',sa,xl,sc,yl,Res)
                if sa >= 1:
                    r[i] = I
                    c[i] = el_f(Na, Nc, xl + 1, yl, sa - 1, sc)
                    D[i] = Res * sa * rhoa  # Sa->Ial
                    i += 1
                    tmp += Res * sa * rhoa
                if xl >= 1:
                    r[i] = I
                    c[i] = el_f(Na, Nc, xl - 1, yl, sa + 1, sc)
                    D[i] = g * xl  # Ial->Sa
                    i += 1
                    tmp += g * xl
                if sc >= 1:
                    r[i] = I
                    c[i] = el_f(Na, Nc, xl, yl + 1, sa, sc - 1)
                    D[i] = Res * sc * rhoc  # Sc->Icl
                    i += 1
                    tmp += Res * sc * rhoc
                if yl >= 1:
                    r[i] = I
                    c[i] = el_f(Na, Nc, xl, yl - 1, sa, sc + 1)
                    D[i] = g * yl  # Icl->Sc
                    i += 1
                    tmp += g * yl
                r[i] = I
                c[i] = I
                D[i] = -tmp
                i += 1
        R = coo_matrix((D, (r, c)), shape=(SIZE, SIZE))
        # print('max',np.max(D))
        R = R.todense()
        self.M = R.T
        return R.T


class SIS_AC_pop(Population):
    '''
    Initialise SIS_AC_pop model class.
    Inputs:
    data (array) - first column  - HH sizes (total)
                    - second column - HH size (adults)
                    - third column - HH size (children)
                    - fourth column (optional) - number Adults infected
                    - fifth column (optional) - number Children infected
    params (tuple)  beta_A - Adult within population transmission rate
                    beta_C - Child within population transmission rate
                    rho_C  - Child susceptibility
                    eps - External FOI
    freq_dep (bool, optional) - frequency dependence
    NOTE: rho_A  - Adult susceptibility - is always 1
    '''

    subclass = SIS_AC
    param_l = subclass.param_l
    risk_l = subclass.risk_l
    dim = len(param_l)
    R = subclass.R

    def data_transform(self, data):
        def el_f(Na, Nc, xl, yl, sa, sc):
            return xl*(Nc+1) + yl

        for i in range(self.infected.shape[0]):
            Na = self.HHsizes[i, 0]
            Nc = self.HHsizes[i, 1]
            xl = data[i, 0]
            yl = data[i, 1]
            self.infected[i] = el_f(Na, Nc, xl, yl, Na-xl, Nc-yl)
        return self.infected

    def __str__(self):
        return 'SIS_AC_pop: m = %d, n = %d' % (self.m, self.N)


class SIS_ACR_homo(Model):
    """
    """

    param_l = ['beta_l', 'beta_h', 'rho_C', 'eps', 'g_h']
    risk_l = ['Na', 'Nc']
    dim = len(param_l)
    sparse = True
    R = 2

    def gen_mat(self):
        '''
        Na, Nc, bl, bh, gl, ra(=1),  rc, gl, eps
        '''
        from snepits._models_spec import SIS_ACR_sp_arr

        N = self.N
        Na = self.Na
        Nc = self.Nc
        bl = self.beta_l
        bh = self.beta_h
        rhoa = 1
        rhoc = self.rho_C
        if self.freq_dep:
            bl *= 1 / (N - 1)
            bh *= 1 / (N - 1)
        eps = self.eps
        gal = 1
        gcl = gal
        gah = self.g_h
        gch = self.g_h

        self.M = SIS_ACR_sp_arr(Na, Nc, bl, bh, bl, bh, gal, gah, 
                                gcl, gch, eps, rhoa, rhoa, rhoc, rhoc)
        return self.M


class SIS_ACR_all(Model):
    """
    """

    param_l = ['beta_al', 'beta_ah', 'beta_cl', 'beta_ch',
               'rho_ah', 'rho_cl', 'rho_ch',
               'g_ah', 'g_cl', 'g_ch', 'eps']
    risk_l = ['Na', 'Nc']
    dim = len(param_l)
    sparse = True
    R = 2

    def gen_mat(self):
        from snepits._models_spec import SIS_ACR_sp_arr

        N = self.N
        Na = self.Na
        Nc = self.Nc

        bal = self.beta_al
        bah = self.beta_ah
        bcl = self.beta_cl
        bch = self.beta_ch
        rhoal = 1
        rhoah = self.rho_ah
        rhocl = self.rho_cl
        rhoch = self.rho_ch
        if self.freq_dep:
            bal *= 1 / (N - 1)
            bah *= 1 / (N - 1)
            bcl *= 1 / (N - 1)
            bch *= 1 / (N - 1)
        eps = self.eps
        gal = 1
        gah = self.g_ah
        gcl = self.g_cl
        gch = self.g_ch

        self.M = SIS_ACR_sp_arr(Na, Nc, bal, bah, bcl, bch, gal, gah, 
                                gcl, gch, eps, rhoal, rhoah, rhocl, rhoch)
        return self.M


class SIS_ACR_all_reparam(Model):
    """
    """

    param_l = ['beta_al', 'beta_ah', 'beta_cl', 'beta_ch',
               'rho_ah', 'rho_cl', 'rho_ch',
               'g_ah', 'g_cl', 'g_ch', 'eps']
    risk_l = ['Na', 'Nc']
    dim = len(param_l)
    sparse = True
    R = 2

    def gen_mat(self):
        from snepits._models_spec import SIS_ACR_sp_arr

        N = self.N
        Na = self.Na
        Nc = self.Nc

        bal = self.beta_al
        bah = self.beta_ah
        bcl = self.beta_cl
        bch = self.beta_ch
        rhoal = 1
        rhoah = self.rho_ah
        rhocl = self.rho_cl
        rhoch = self.rho_ch
        if self.freq_dep:
            bal *= 1 / (N - 1)
            bah *= 1 / (N - 1)
            bcl *= 1 / (N - 1)
            bch *= 1 / (N - 1)
        eps = self.eps
        gal = 1
        gah = self.rho_ah * self.g_ah
        gcl = self.rho_cl * self.g_cl
        gch = self.rho_ch * self.g_ch

        self.M = SIS_ACR_sp_arr(Na, Nc, bal, bah, bcl, bch, gal, gah, 
                                gcl, gch, eps, rhoal, rhoah, rhocl, rhoch)
        return self.M


class SIS_ACR_orthogonal_reparam(Model):
    """
    """

    param_l = ['beta_c', 'beta_l', 'beta_h',
               'rho_c', 'rho_h',
               'g', 'eps']
               # 'g_cl', 'g_ch', 'g_h', 'eps']
    risk_l = ['Na', 'Nc']
    dim = len(param_l)
    sparse = True
    R = 2

    def gen_mat(self):
        from snepits._models_spec import SIS_ACR_sp_arr

        N = self.N
        Na = self.Na
        Nc = self.Nc

        bal = self.beta_l
        bah = self.beta_h
        bcl = self.beta_c * self.beta_l
        bch = self.beta_c * self.beta_h
        rhoal = 1
        rhoah = self.rho_h
        rhocl = self.rho_c
        rhoch = self.rho_c * self.rho_h
        if self.freq_dep:
            bal *= 1 / (N - 1)
            bah *= 1 / (N - 1)
            bcl *= 1 / (N - 1)
            bch *= 1 / (N - 1)
        eps = self.eps
        gal = 1
        gah = self.g#self.g_h * self.rho_h
        gcl = 1 #self.g_cl * self.rho_c * self.rho_l
        gch = self.g#self.g_ch * self.rho_c * self.rho_h

        self.M = SIS_ACR_sp_arr(Na, Nc, bal, bah, bcl, bch, gal, gah,
                                gcl, gch, eps, rhoal, rhoah, rhocl, rhoch)
        return self.M


class SIS_ACR_original(Model):
    """
    """

    param_l = ['beta_Al', 'beta_Ah', 'beta_Cl', 'beta_Ch',
            'rho_C', 'eps', 'g_h', 'alpha']
    risk_l = ['Na', 'Nc']
    dim = len(param_l)
    sparse = True
    R = 2

    def gen_mat(self):
        from snepits._models_spec import SIS_ACR_sp_arr

        N = self.N
        Na = self.Na
        Nc = self.Nc

        bal = self.beta_Al
        bah = self.beta_Ah
        bcl = self.beta_Cl
        bch = self.beta_Ch
        rhoal = 1
        rhoah = 1
        rhocl = self.rho_C
        rhoch = self.rho_C
        alpha = self.alpha
        if N > 1:
            bal *= 1 / (N - 1)**(alpha)
            bah *= 1 / (N - 1)**(alpha)
            bcl *= 1 / (N - 1)**(alpha)
            bch *= 1 / (N - 1)**(alpha)
        eps = self.eps
        gal = 1
        gah = self.g_h
        gcl = 1
        gch = self.g_h

        self.M = SIS_ACR_sp_arr(Na, Nc, bal, bah, bcl, bch, gal, gah,
                                gcl, gch, eps, rhoal, rhoah, rhocl, rhoch)
        return self.M


class SIS_ACR_original_2(Model):
    """
    """

    param_l = ['beta_Al', 'beta_Ah', 'beta_Cl', 'beta_Ch',
            'rho_C', 'eps', 'g_h']
    risk_l = ['Na', 'Nc']
    dim = len(param_l)
    sparse = True
    R = 2

    def gen_mat(self):
        from snepits._models_spec import SIS_ACR_sp_arr

        N = self.N
        Na = self.Na
        Nc = self.Nc

        bal = self.beta_Al
        bah = self.beta_Ah
        bcl = self.beta_Cl
        bch = self.beta_Ch
        rhoal = 1
        rhoah = 1
        rhocl = self.rho_C
        rhoch = self.rho_C
        if self.freq_dep and N > 1:
            bal *= 1 / (N - 1)
            bah *= 1 / (N - 1)
            bcl *= 1 / (N - 1)
            bch *= 1 / (N - 1)
        eps = self.eps
        gal = 1
        gah = self.g_h
        gcl = 1
        gch = self.g_h

        self.M = SIS_ACR_sp_arr(Na, Nc, bal, bah, bcl, bch, gal, gah,
                                gcl, gch, eps, rhoal, rhoah, rhocl, rhoch)
        return self.M


class SIS_ACR_pop(Population):
    '''
    Args:
        data (array): Columns:
                       - HH sizes (total)
                       - HH size (adults)
                       - HH size (children)
                       - Adults lightly infected
                       - Adults heaviliy infected
                       - Children lightly infected
                       - Children heavily infected
        params (tuple):  TODO
        freq_dep (bool, optional): frequency dependence

    NOTE: rho_A  - Adult susceptibility - is always 1
    '''

    R = 2

    def __str__(self):
        return 'SIS_ACR_pop: m = %d, n = %d' % (self.m, self.N)

    def data_transform(self, data):
        from snepits._models_spec import el_f

        for i in range(self.infected.shape[0]):
            Na = self.HHsizes[i, 0]
            Nc = self.HHsizes[i, 1]
            xl = data[i, 0]
            xh = data[i, 1]
            yl = data[i, 2]
            yh = data[i, 3]
            self.infected[i] = el_f(Na, Nc, xl, yl,
                                    Na - xl - xh, Nc - yl - yh)
        return self.infected


class SIS_ACR_pop_gen(SIS_ACR_pop):
    """ Generates a population class corresponding to `subclass`

    Avoids having to specify an identical population class for differing
    ACR models.
    """

    def __init__(self, subclass, *args, **kwargs):
        self.subclass = subclass
        self.R = subclass.R
        self.param_l = subclass.param_l
        self.risk_l = subclass.risk_l
        self.dim = len(self.param_l)
        super().__init__(*args, **kwargs)
