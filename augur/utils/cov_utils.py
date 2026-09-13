import logging
import numpy as np
import pyccl as ccl
from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky
from augur.utils.config_io import parse_array
from augur.generate_utils.cmb_lensing import CMB_TRACER_NAME, get_cmb_noise

logger = logging.getLogger(__name__)


def get_noise_power(config, S, tracer_name, return_ndens=False):
    """ Returns noise power for tracer

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config

    S : Sacc
        Sacc file containing all tracers

    tracer_name : str
        Tracer ID

    return_ndens : bool
        If `True` return the number density (in arcmin^-2)

    Returns:
    -------
    noise : float
       Noise power for Cls for that particular tracer. That is 1/nbar for
       number counts and e**2/nbar for weak lensing tracer.

    Note:
    -----
    The input number_densities are #/arcmin. The output noise has no units.
    The output number density is in arcmin^-2.
    """
    if 'src' not in tracer_name and 'lens' not in tracer_name:
        logger.error("Cannot compute noise for source of kind %s.", tracer_name[:-1])
        raise NotImplementedError
    nz_all = dict()
    nz_all['src'] = []
    nz_all['lens'] = []
    for tr in S.tracers:
        trobj = S.get_tracer(tr)
        # obtain some arbitrary number of bins
        prefix = tr.rstrip('0123456789')
        # Skip tracers that carry no n(z) -- e.g. a CMB convergence Map tracer.
        # This loop runs over every tracer in the file regardless of which one
        # was asked about, so without this guard the mere presence of such a
        # tracer breaks get_noise_power for `src`/`lens` too.
        if prefix not in nz_all:
            continue
        nz_all[prefix].append(trobj.nz)
    tracer_prefix = tracer_name.rstrip('0123456789')
    tracer_bin_str = tracer_name[len(tracer_prefix):]
    if tracer_bin_str == '':
        raise ValueError(
            f"Tracer name '{tracer_name}' must end with a bin index."
        )
    tracer_bin = int(tracer_bin_str)

    norm = dict()
    if 'src' in tracer_name:
        nz_all['src'] = np.array(nz_all['src'])
        norm['src'] = np.sum(nz_all['src'], axis=1)/np.sum(nz_all['src'])

        ndens = config['sources']['ndens']
        ndens *= norm['src'][tracer_bin]
    elif 'lens' in tracer_name:
        nz_all['lens'] = np.array(nz_all['lens'])
        norm['lens'] = np.sum(nz_all['lens'], axis=1)/np.sum(nz_all['lens'])
        ndens = config['lenses']['ndens']
        ndens *= norm['lens'][tracer_bin]
    nbar = ndens * (180 * 60 / np.pi) ** 2  # per steradian

    if 'src' in tracer_name:
        noise_power = config['sources']['ellipticity_error'] ** 2 / nbar
    elif 'lens' in tracer_name:
        noise_power = 1 / nbar
    else:
        logger.error("Cannot do error for source of kind %s.", tracer_prefix)
        raise NotImplementedError
    if return_ndens:
        return noise_power, ndens
    else:
        return noise_power


def get_gaus_cov(S, lk, cosmo, fsky, config):
    """
    Basic implementation of Gaussian covariance using the mode-counting formula
    and fsky approximation. Assumes that all ell-edges are the same for (3x)2pt
    statistics and uniform bin-widths in ell.

    Parameters:
    -----------
    S : Sacc
        Sacc object containing the data-vector for which we want the covariance
    lk : firecrown.likelihood
        Likelihood object containing the statistics for which we want to compute
        the covariance matrix.
    cosmo : ccl.Cosmology object
        Fiducial cosmology in which to evaluate the covariance matrix.
    fsky : float
        Fraction of the sky observed.
    config : dict
        Configuration dictionary

    Returns:
    --------
    cov_all : np.ndarray
        Numpy array containing the covariance matrix.
    """
    # Initialize big matrix
    cov_all = np.zeros((len(S.data), len(S.data)))

    ell_edges_by_stat = {
        stat_name: np.asarray(parse_array(stat_cfg['ell_edges']))
        for stat_name, stat_cfg in config['statistics'].items()
    }
    # CMB lensing statistics live in their own config section but share the
    # same ell-edge bookkeeping.
    cmb_cfg = config.get('cmb_lensing', None)
    if cmb_cfg is not None:
        for stat_name, stat_cfg in cmb_cfg.get('statistics', {}).items():
            ell_edges_by_stat[stat_name] = np.asarray(parse_array(stat_cfg['ell_edges']))

    def _noise_between(tr_a_name, tr_b_name, ells):
        """Uncorrelated noise term N_ab (non-zero only for auto-tracer pairs)."""
        if tr_a_name != tr_b_name:
            return 0.0
        if tr_a_name == CMB_TRACER_NAME:
            if cmb_cfg is None:
                raise ValueError(
                    f"SACC contains the tracer '{CMB_TRACER_NAME}' but the config "
                    "has no `cmb_lensing` section to take its reconstruction "
                    "noise from."
                )
            return get_cmb_noise(cmb_cfg, ells)
        return get_noise_power(config, S, tr_a_name)

    # Loop over statistic in the likelihood (assuming 3x2pt so far)
    for i, myst1 in enumerate(lk.statistics):
        myst1 = myst1.statistic
        tr1_name = myst1.source0.sacc_tracer
        tr2_name = myst1.source1.sacc_tracer
        tr1 = myst1.source0.tracers[0].ccl_tracer  # Pulling out the tracers
        tr2 = myst1.source1.tracers[0].ccl_tracer
        ell12 = myst1.ells
        # Loop over upper-triangle and fill lower-triangle by symmetry
        for j in range(i, len(lk.statistics)):
            myst2 = lk.statistics[j]
            myst2 = myst2.statistic
            tr3_name = myst2.source0.sacc_tracer
            tr4_name = myst2.source1.sacc_tracer
            tr3 = myst2.source0.tracers[0].ccl_tracer
            tr4 = myst2.source1.tracers[0].ccl_tracer
            ell34 = myst2.ells
            # Assuming that everything has the same ell-edges and we are just changing the length
            # TODO update this for a more general case
            if len(ell34) < len(ell12):
                ells_here = ell34
                stat_here = myst2.sacc_data_type
            else:
                ells_here = ell12
                stat_here = myst1.sacc_data_type
            # Get the necessary Cls
            cls13 = ccl.angular_cl(cosmo, tr1, tr3, ells_here)
            cls24 = ccl.angular_cl(cosmo, tr2, tr4, ells_here)
            cls14 = ccl.angular_cl(cosmo, tr1, tr4, ells_here)
            cls23 = ccl.angular_cl(cosmo, tr2, tr3, ells_here)

            # Add uncorrelated noise (shot/shape) terms for auto-tracer pairs:
            # C_ab -> C_ab + N_ab, where N_ab != 0 only if a == b.
            cls13 += _noise_between(tr1_name, tr3_name, ells_here)
            cls24 += _noise_between(tr2_name, tr4_name, ells_here)
            cls14 += _noise_between(tr1_name, tr4_name, ells_here)
            cls23 += _noise_between(tr2_name, tr3_name, ells_here)

            # Normalization factor
            dell = np.diff(ell_edges_by_stat[stat_here])[:len(ells_here)]
            norm = dell*(2*ells_here+1)*fsky
            cov_here = cls13*cls24 + cls14*cls23
            cov_here /= norm
            # The following lines only work if the ell-edges are constant across the probes
            # and we just vary the length
            n_ells = min(len(ell12), len(ell34))
            # Use the sacc indices to write the matrix in the correct order
            cov_all[myst1.sacc_indices[:n_ells], myst2.sacc_indices[:n_ells]] = cov_here[:n_ells]
            cov_all[myst2.sacc_indices[:n_ells], myst1.sacc_indices[:n_ells]] = cov_here[:n_ells]
    return cov_all


def get_SRD_cov(config, S):
    """
    Read covariance file from SRD v1:
    https://github.com/LSSTDESC/Requirements/tree/master/forecasting/WL-LSS-CL/cov

    Parameters:
    -----------
    config : dict
        Dictionary containing the relevant two_point section of the config.
    S : sacc.Sacc
        Sacc object containing the data vector for which to compute the covariance.

    Returns:
    --------
    cov_all : np.ndarray
        Numpy array containing the SRD covariance matrix with the relevant shape as established
        by the input sacc file S.
    """
    if 'SRD_cov_path' not in config.keys():
        raise ValueError('SRD_cov_path is needed to use SRD covariance.')
    cov_in = np.load(config['SRD_cov_path'])
    ncls = 20
    # Data combinations for Y1 as per SRD v1
    data_combs_y1 = [('src0', 'src0'), ('src0', 'src1'), ('src0', 'src2'), ('src0', 'src3'),
                     ('src0', 'src4'), ('src1', 'src1'), ('src1', 'src2'), ('src1', 'src3'),
                     ('src1', 'src4'), ('src2', 'src2'), ('src2', 'src3'), ('src2', 'src4'),
                     ('src3', 'src3'), ('src3', 'src4'), ('src4', 'src4'),
                     ('src2', 'lens0'), ('src3', 'lens0'), ('src4', 'lens0'),
                     ('src3', 'lens1'), ('src4', 'lens1'), ('src4', 'lens2'),
                     ('src4', 'lens3'), ('lens0', 'lens0'), ('lens1', 'lens1'),
                     ('lens2', 'lens2'), ('lens3', 'lens3'), ('lens4', 'lens4')]
    # Data combinations for Y10 as per SRD v1
    data_combs_y10 = [('src0', 'src0'), ('src0', 'src1'), ('src0', 'src2'), ('src0', 'src3'),
                      ('src0', 'src4'), ('src1', 'src1'), ('src1', 'src2'), ('src1', 'src3'),
                      ('src1', 'src4'), ('src2', 'src2'), ('src2', 'src3'), ('src2', 'src4'),
                      ('src3', 'src3'), ('src3', 'src4'), ('src4', 'src4'),
                      ('src1', 'lens0'), ('src2', 'lens0'), ('src3', 'lens0'), ('src4', 'lens0'),
                      ('src1', 'lens1'), ('src2', 'lens1'), ('src3', 'lens1'), ('src4', 'lens1'),
                      ('src2', 'lens2'), ('src3', 'lens2'), ('src4', 'lens2'),
                      ('src2', 'lens3'), ('src3', 'lens3'), ('src4', 'lens3'),
                      ('src2', 'lens4'), ('src3', 'lens4'), ('src4', 'lens4'),
                      ('src3', 'lens5'), ('src4', 'lens5'),
                      ('src3', 'lens6'), ('src4', 'lens6'),
                      ('src3', 'lens7'), ('src4', 'lens7'),
                      ('src4', 'lens8'),
                      ('src4', 'lens9'),
                      ('lens0', 'lens0'), ('lens1', 'lens1'), ('lens2', 'lens2'),
                      ('lens3', 'lens3'), ('lens4', 'lens4'), ('lens5', 'lens5'),
                      ('lens6', 'lens6'), ('lens7', 'lens7'), ('lens8', 'lens8'),
                      ('lens9', 'lens9')]

    if 'Y10' in config['SRD_cov_path']:
        data_combs = data_combs_y10
    else:
        data_combs = data_combs_y1

    cov_sacc_all = np.zeros((len(S.mean), len(S.mean)))
    for i, comb1 in enumerate(data_combs):
        dtype_here1 = S.get_data_types(tracers=comb1)[0]
        inds1 = S.indices(data_type=dtype_here1, tracers=comb1)
        for j, comb2 in enumerate(data_combs):
            dtype_here2 = S.get_data_types(tracers=comb2)[0]
            inds2 = S.indices(data_type=dtype_here2, tracers=comb2)
            inds_all = np.meshgrid(inds1, inds2)
            cov_sacc_all[inds_all[0].T, inds_all[1].T] = cov_in[ncls*i:ncls*i+len(inds1),
                                                                ncls*j:ncls*j+len(inds2)]
    return cov_sacc_all


class TJPCovGaus(FourierGaussianFsky):
    """
    Class to patch FourierGaussianFsky to work with Augur
    """

    def __init__(self, config):
        super().__init__(config)
        # self.tracer_Noise = self.tracer_Noise_coupled

    def get_tracer_info(self, return_noise_coupled=False):
        """Return tracer info, supplying the CMB convergence noise TJPCov omits.

        TJPCov builds a ``CMBLensingTracer`` for ``quantity == 'cmb_convergence'``
        but never assigns it a ``tracer_Noise`` entry, so
        ``FourierGaussianFsky.get_covariance_block`` raises ``KeyError``. That
        hits 5x2pt as well as 6x2pt, because the guard compares the *first*
        tracer of each pair and so the ``(kappa, lens_i) x (kappa, lens_j)``
        cross-block trips it even with no kappa auto-spectrum present.

        TEMPORARY. The real fix belongs upstream in TJPCov
        (https://github.com/LSSTDESC/TJPCov/issues/124); remove this override
        once that lands rather than leaving CMB-lensing knowledge about another
        package sitting in augur.

        ``tjpcov.cmb_noise`` may be a scalar or an array on the binning grid.
        """
        ccl_tracers, tracer_Noise, tracer_Noise_coupled = super().get_tracer_info(
            return_noise_coupled=True
        )

        cmb_noise_cfg = self.config['tjpcov'].get('cmb_noise', 0.0)
        ell, _, _ = self.get_binning_info()
        if isinstance(cmb_noise_cfg, dict):
            # A `cmb_lensing` config section: evaluate the curve on TJPCov's own
            # integration grid, which is the only way to be sure of the length.
            cmb_noise = get_cmb_noise(cmb_noise_cfg, ell)
        else:
            cmb_noise = np.asarray(cmb_noise_cfg, dtype=float)
            if cmb_noise.ndim > 1:
                raise ValueError(
                    "`tjpcov.cmb_noise` must be a scalar, a 1D array or a "
                    f"`cmb_lensing` config dict, got shape {cmb_noise.shape}."
                )
            if cmb_noise.ndim == 1 and cmb_noise.size != len(ell):
                # A silent length mismatch would broadcast into a wrong
                # covariance rather than fail, so check it here.
                raise ValueError(
                    f"`tjpcov.cmb_noise` has {cmb_noise.size} entries but "
                    f"TJPCov integrates over {len(ell)} ells. Pass the "
                    "`cmb_lensing` config section instead and let it be "
                    "evaluated on the right grid."
                )
            if cmb_noise.ndim == 0:
                cmb_noise = float(cmb_noise)

        # Outside the reconstruction band get_cmb_noise returns inf, which is
        # the honest answer but would make the covariance non-invertible. Give
        # those modes a large finite noise instead: the weight is ~0 either way.
        if np.ndim(cmb_noise) > 0 and not np.all(np.isfinite(cmb_noise)):
            finite = cmb_noise[np.isfinite(cmb_noise)]
            if finite.size == 0:
                raise ValueError(
                    "The CMB lensing noise curve is infinite across the whole "
                    "TJPCov ell range; check the reconstruction band against "
                    "`tjpcov.binning_info.ell_edges`."
                )
            big = 1e6 * float(np.max(finite))
            n_bad = int(np.sum(~np.isfinite(cmb_noise)))
            logger.warning(
                "%d of %d ells lie outside the CMB lensing reconstruction band; "
                "their noise is set to %g so the covariance stays invertible.",
                n_bad, cmb_noise.size, big,
            )
            cmb_noise = np.where(np.isfinite(cmb_noise), cmb_noise, big)

        sacc_file = self.io.get_sacc_file()
        for tracer in sacc_file.tracers:
            tracer_dat = sacc_file.get_tracer(tracer)
            if getattr(tracer_dat, 'quantity', None) == 'cmb_convergence':
                tracer_Noise.setdefault(tracer, cmb_noise)

        self.ccl_tracers = ccl_tracers
        self.tracer_Noise = tracer_Noise
        self.tracer_Noise_coupled = tracer_Noise_coupled

        if return_noise_coupled:
            return ccl_tracers, tracer_Noise, tracer_Noise_coupled
        return ccl_tracers, tracer_Noise

    def get_binning_info(self):
        ell_eff = self.get_ell_eff()
        if 'ell_edges' not in self.config['tjpcov']['binning_info'].keys():
            raise ValueError(
                "`tjpcov.binning_info.ell_edges` must be defined in the configuration "
                "when using TJPCov with Augur."
            )
        ell_edges = self.config['tjpcov']['binning_info']['ell_edges']
        ell_min = np.min(ell_edges)
        ell_max = np.max(ell_edges)
        nbpw = int(ell_max - ell_min)
        ell = np.linspace(ell_min, ell_max, nbpw+1).astype(np.int32)
        return ell, ell_eff, ell_edges
