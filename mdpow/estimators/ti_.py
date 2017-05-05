import alchemlyb.estimators
import numpy as np
import pandas as pd
import numkit

class TI(alchemlyb.estimators.TI):
    """Thermodynamic integration (TI) estimator

    Parameters
    ----------

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.
    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences.
    delta_f_error_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences while taking correlated data
        into account.
    """

    def fit(self, dHdl):
        """
        Compute free energy differences between each state by integrating
        dHdl across lambda values.

        Parameters
        ----------
        dHdl : DataFrame
            dHdl[n,k] is the potential energy gradient with respect to lambda
            for each configuration n and lambda k.

        """

        # sort by state so that rows from same state are in contiguous blocks,
        # and adjacent states are next to each other
        dHdl = dHdl.sort_index(level=dHdl.index.names[1:])

        # obtain the mean and variance of the mean for each state
        # variance calculation assumes no correlation between points
        # used to calculate mean
        means = dHdl.mean(level=dHdl.index.names[1:])
        variances = np.square(dHdl.sem(level=dHdl.index.names[1:]))
        errors = self._correlated_error(dHdl)

        # obtain vector of delta lambdas between each state
        dl = means.reset_index()[means.index.names[:]].diff().iloc[1:].values

        # apply trapezoid rule to obtain DF between each adjacent state
        deltas = (dl * (means.iloc[:-1].values + means.iloc[1:].values)/2).sum(axis=1)
        d_deltas = (dl**2 * (variances.iloc[:-1].values + variances.iloc[1:].values)/4).sum(axis=1)
        d_deltas_errors = (dl**2 * (errors.iloc[:-1].values + errors.iloc[1:].values)/4).sum(axis=1)

        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas)+1, len(deltas)+1))
        ad_delta = np.zeros_like(adelta)
        ad_delta_errors = np.zeros_like(adelta)

        for j in range(len(deltas)):
            out = []
            dout = []
            douterror = []
            for i in range(len(deltas) - j):
                out.append(deltas[i] + deltas[i+1:i+j+1].sum())
                dout.append(d_deltas[i] + d_deltas[i+1:i+j+1].sum())
                douterror.append(d_deltas_errors[i] + d_deltas_errors[i+1:i+j+1].sum())

            adelta += np.diagflat(np.array(out), k=j+1)
            ad_delta += np.diagflat(np.array(dout), k=j+1)
            ad_delta_errors += np.diagflat(np.array(douterror), k=j+1)

        # yield standard delta_f_ free energies between each state
        self.delta_f_ = pd.DataFrame(adelta - adelta.T,
                                     columns=means.index.values,
                                     index=means.index.values)

        # yield standard deviation d_delta_f_ between each state
        self.d_delta_f_ = pd.DataFrame(np.sqrt(ad_delta + ad_delta.T),
                                       columns=variances.index.values,
                                       index=variances.index.values)

        self.d_delta_f_error_ = pd.DataFrame(np.sqrt(ad_delta_errors + ad_delta_errors.T),
                                       columns=errors.index.values,
                                       index=errors.index.values)


        return self

    def _correlated_error(dHdl):
        """Compute errors considering correlated data from time series.

        Parameters
        ----------
        dHdl : DataFrame
            dHdl is the time series of dHdl for a particular lambda window.

        Returns
        -------
        errors of correlated data
        """

        errors = []
        lambdas = []
        for name, group in dHdl.groupby(level='fep-lambda'):
            tc = numkit.timeseries.tcorrel(group.index.get_level_values('time').values,
                                           group.values.flatten())
            lambdas.append(name)
            errors.append(tc['sigma'])

        return pd.DataFrame(errors,
                            index=pd.Float64Index(lambdas, name='fep-lambda'),
                            columns=['fep'])
