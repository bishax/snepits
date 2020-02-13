import logging

import snepits
from snepits.models.models_meta import Model, Population

logger = logging.getLogger(__name__)


class SIS_A2R3(Model):

    param_l = ['beta_l', 'beta_m', 'beta_h', 'beta_C',
               'rho_m', 'rho_h', 'rho_C', 'eps', 'g_m', 'g_h', 'g_C', 'logit_alpha']
    risk_l = ['Na', 'Nc']
    sparse = True
    R = 3

    def gen_mat(self):
        from snepits._models_spec import SIS_A2R3_rho
        from scipy.special import expit

        N = self.N
        Na = self.Na
        Nc = self.Nc
        bal = self.beta_l
        bam = self.beta_m
        bah = self.beta_h
        bcl = self.beta_l * self.beta_C
        bcm = self.beta_m * self.beta_C
        bch = self.beta_h * self.beta_C
        rhoal = 1
        rhoam = self.rho_m
        rhoah = self.rho_h
        rhocl = self.rho_C
        rhocm = self.rho_m * self.rho_C
        rhoch = self.rho_h * self.rho_C
        alpha = expit(self.logit_alpha)
        bal *= 1 / (N - 1)**alpha
        bcl *= 1 / (N - 1)**alpha
        bam *= 1 / (N - 1)**alpha
        bcm *= 1 / (N - 1)**alpha
        bah *= 1 / (N - 1)**alpha
        bch *= 1 / (N - 1)**alpha
        eps = self.eps
        gal = 1
        gam = self.g_m
        gah = self.g_h
        gcl = self.g_C
        gcm = self.g_m * self.g_C
        gch = self.g_h * self.g_C
        self.M = SIS_A2R3_rho(Na, Nc, bal, bam, bah, bcl, bcm, bch, gal, gam, gah, gcl, gcm, gch, eps, rhoal, rhoam, rhoah, rhocl, rhocm, rhoch, 10)
        return self.M


class SIS_A2R3_pop(Population):
    '''
    Initialise SIS_ACR_pop model class.
    Inputs:
    data (array) - first column  - HH sizes (total)
                    - second column - HH size (adults)
                    - third column - HH size (children)
                    - fourth column (optional) - number Adults lightly infected
                    - fifth column (optional) - number Adults heaviliy infected
                    - sixth column (optional) - number Children lightly infected
                    - seventh column (optional) - number Children heavily infected
    params (tuple)  TODO

    NOTE: rho_A  - Adult susceptibility - is always 1
    '''
    subclass = SIS_A2R3
    param_l = subclass.param_l
    risk_l = subclass.risk_l
    R = subclass.R

    def calc_prior(self):
        from numpy import inf, array
        if self.beta_m > self.beta_h:
            return -inf
        elif self.beta_l > self.beta_m:
            return -inf
        elif (array(self.params) > 100).sum() > 1:
            return -inf
        else:
            return 0

    def __str__(self):
        return 'SIS_A2R3_pop: m = %d, n = %d' % (self.m, self.N)


class SIS_A2R3_fd(Model):

    param_l = ['beta_l', 'beta_m', 'beta_h', 'beta_C',
               'rho_m', 'rho_h', 'rho_C', 'eps', 'g_m', 'g_h', 'g_C']
    dim = len(param_l)
    risk_l = ['Na', 'Nc']
    sparse = True
    R = 3

    def gen_mat(self):
        from snepits._models_spec import SIS_A2R3_rho

        N = self.N
        Na = self.Na
        Nc = self.Nc
        bal = self.beta_l
        bam = self.beta_m
        bah = self.beta_h
        bcl = self.beta_l * self.beta_C
        bcm = self.beta_m * self.beta_C
        bch = self.beta_h * self.beta_C
        rhoal = 1
        rhoam = self.rho_m
        rhoah = self.rho_h
        rhocl = self.rho_C
        rhocm = self.rho_m * self.rho_C
        rhoch = self.rho_h * self.rho_C
        bal *= 1 / (N - 1)
        bcl *= 1 / (N - 1)
        bam *= 1 / (N - 1)
        bcm *= 1 / (N - 1)
        bah *= 1 / (N - 1)
        bch *= 1 / (N - 1)
        eps = self.eps
        gal = 1
        gam = self.g_m
        gah = self.g_h
        gcl = self.g_C
        gcm = self.g_m * self.g_C
        gch = self.g_h * self.g_C
        self.M = SIS_A2R3_rho(Na, Nc, bal, bam, bah, bcl, bcm, bch, gal, gam,
                gah, gcl, gcm, gch, eps, rhoal, rhoam, rhoah, rhocl, rhocm,
                rhoch, 12)
        return self.M


class SIS_A2R3_fd_pop(Population):
    '''
    Initialise SIS_ACR_pop model class.
    Inputs:
    data (array) - first column  - HH sizes (total)
                    - second column - HH size (adults)
                    - third column - HH size (children)
                    - fourth column (optional) - number Adults lightly infected
                    - fifth column (optional) - number Adults heaviliy infected
                    - sixth column (optional) - number Children lightly infected
                    - seventh column (optional) - number Children heavily infected
    params (tuple)  TODO

    NOTE: rho_A  - Adult susceptibility - is always 1
    '''
    subclass = SIS_A2R3_fd
    param_l = subclass.param_l
    risk_l = subclass.risk_l
    dim = subclass.dim
    R = subclass.R

    def calc_prior(self):
        from numpy import inf, array
        if self.beta_m > self.beta_h:
            return -inf
        elif self.beta_l > self.beta_m:
            return -inf
        elif (array(self.params) > 100).sum() > 1:
            return -inf
        else:
            return 0

    def data_transform(self, data):
        from _models_spec import el_f_A2_R3

        for i in range(self.infected.shape[0]):
            Na = self.HHsizes[i, 0]
            Nc = self.HHsizes[i, 1]
            xl = data[i, 0]
            xm = data[i, 1]
            xh = data[i, 2]
            yl = data[i, 3]
            ym = data[i, 4]
            yh = data[i, 5]
            self.infected[i] = el_f_A2_R3(
                    Na,
                    Nc,
                    xl,
                    xm,
                    yl,
                    ym,
                    Na - xl - xm - xh,
                    Nc - yl - ym - yh,
                    )
        return self.infected

    def __str__(self):
        return 'SIS_A2R3_fd_pop: m = %d, n = %d' % (self.m, self.N)


class SIS_A3R3(Model):

    param_l = ['beta_l', 'beta_m', 'beta_h', 'beta_I', 'beta_C',
               'rho_m', 'rho_h', 'eps', 'g_m', 'g_h', 'alpha']
    dim = len(param_l)
    risk_l = ['Na', 'Nc', 'Ni']
    sparse = True
    R = 3

    def gen_mat(self):
        from snepits.pyovpyx import SIS_A3R3_rho

        N = self.N
        Na = self.Na
        Nc = self.Nc
        Ni = self.Ni
        bal = self.beta_l
        bam = self.beta_m
        bah = self.beta_h
        bcl = self.beta_l * self.beta_C
        bcm = self.beta_m * self.beta_C
        bch = self.beta_h * self.beta_C
        bil = self.beta_l * self.beta_I
        bim = self.beta_m * self.beta_I
        bih = self.beta_h * self.beta_I
        rhoal = 1
        rhoam = self.rho_m
        rhoah = self.rho_h
        rhocl = 1
        rhocm = self.rho_m
        rhoch = self.rho_h
        rhoil = 1
        rhoim = self.rho_m
        rhoih = self.rho_h
        alpha = self.alpha
        bal *= 1 / (N - 1)**alpha
        bcl *= 1 / (N - 1)**alpha
        bam *= 1 / (N - 1)**alpha
        bcm *= 1 / (N - 1)**alpha
        bah *= 1 / (N - 1)**alpha
        bch *= 1 / (N - 1)**alpha
        eps = self.eps
        gal = 1
        gam = self.g_m
        gah = self.g_h
        gcl = 1
        gcm = self.g_m
        gch = self.g_h
        gil = 1
        gim = self.g_m
        gih = self.g_h

        self.M = SIS_A3R3_rho(Na, Nc, Ni,
                bal, bam, bah, bcl, bcm, bch, bil, bim, bih,
                gal, gam, gah, gcl, gcm, gch, gil, gim, gih,
                eps,
                rhoal, rhoam, rhoah, rhocl, rhocm, rhoch, rhoil, rhoim, rhoih,
                15)
        return self.M


class SIS_A3R3_pop(Population):
    '''
    '''
    subclass = SIS_A3R3
    param_l = subclass.param_l
    risk_l = subclass.risk_l
    dim = subclass.dim
    R = subclass.R

    def data_transform(self, data):
        from snepits.pyovpyx import el_f_A3_R3

        for i in range(self.infected.shape[0]):
            Na = self.HHsizes[i, 0]
            Nc = self.HHsizes[i, 1]
            Ni = self.HHsizes[i, 2]
            xl = data[i, 0]
            xm = data[i, 1]
            xh = data[i, 2]
            yl = data[i, 3]
            ym = data[i, 4]
            yh = data[i, 5]
            zl = data[i, 6]
            zm = data[i, 7]
            zh = data[i, 8]
            self.infected[i] = el_f_A3_R3(
                    Na, Nc, Ni,
                    xl, xm,
                    yl, ym,
                    zl, zm,
                    Na - xl - xm - xh,
                    Nc - yl - ym - yh,
                    Ni - zl - zm - zh,
                    )
        return self.infected
