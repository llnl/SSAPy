"""
Classes for sampling orbit model parameters.
"""

import numpy as np
from .orbit import Orbit as _Orbit
from .compute import rv
from . import utils


class Particles:
    """A class for importance (re-)sampling orbit model parameter samples

    Parameters
    ----------
    particles : (num_particles, 6) array_like
        Positions and velocities of orbiting object at epoch in meters and meters / second.
        The 'chain' part of the output from rvsampler.XXSampler.sample
    rvprobability : class instance
        An instance of an RVProbability for this epoch. The Particles class wraps the `lnlike`
        method of the supplied RVProbability object.
    ln_weights : (num_particles,) array_like, optional
        Weights for each particle in the input particles

    Attributes
    ----------
    particles : (num_particles, ) array_like
        Positions and velocities of orbiting object at epoch in meters and meters / second.
        The 'chain' part of the output from RVSampler.sample
    rvprobability : class instance
        An instance of an RVProbability for this epoch. The Particles class wraps the `lnlike`
        method of the supplied RVProbability object.
    ln_wts : (num_particles,) array_like
        Log weights for each particle.
    num_particles : int
        Number of particles
    initial_particles : (num_particles, 6) array_like
        Copy of input particles to enable reset_to_pseudo_prior()
    initial_ln_wts : (num_particles,) array_like
        Copy of input log weights to enable reset_to_pseudo_prior()
    epoch : float or astropy.time.Time
        The time at which the model parameters (position and velocity) are specified. If float,
        then should correspond to GPS seconds; i.e., seconds since 1980-01-06 00:00:00 UTC.
    orbits : list
        List of Orbit class instances derived from each particle
    lnpriors : (num_particles,) array_like
        Log prior probabilities for each particle. The priors are as set in the input RVProbability
        object.

    Methods
    -------
    reset_to_pseudo_prior()
        Reset the particles and weights to their values at instantiation
    move(epoch)
        Move particles to the specified epoch
    lnlike(epoch_particles)
        Evaluate the log-likelihood of the input particles given internally stored measurements
    reweight(epoch_particles)
        Reweight particles using cross-epoch likelihoods
    resample(num_particles)
        Resample particles to achieve near-uniform weighting per particle
    fuse(epoch_particles, verbose)
        Fuse particles from a different epoch and fuse internal particles and weights
    draw_orbit()
        Draw a single orbit model from the internal list of particles
    mean()
        Evaluate the weighted mean of the particle values
    """
    def __init__(self,
                 particles,
                 rvprobability,
                 lnpriors=None,
                 ln_weights=None):
        self._orbits = None
        self._lnpriors = None
        # Check if we have 3D array of particles from RVSampler.
        # If so, flatten into 2D array of (samples, parameters)
        if particles.ndim == 3:
            particles = np.vstack(particles)
            if lnpriors is not None:
                if lnpriors.ndim != 2:
                    raise ValueError("Supplied lnpriors do not match expected shape from particles")
                self._lnpriors = lnpriors.ravel()
        self.particles = particles
        self.rvprobability = rvprobability
        if ln_weights is None:
            # ln_weights = np.zeros(particles.shape[0], dtype=np.float64)
            # Subtract the log-prior probabilities to approximate samples from the likelihood
            ln_weights = -self.lnpriors
        self.ln_wts = ln_weights
        self.num_particles = self.particles.shape[0]
        #
        # Save a copy of the particles and weights to enable resets to
        # pseudo-prior
        #
        self.initial_particles = particles.copy()
        self.initial_ln_wts = ln_weights.copy()

    def __repr__(self):
        pmean = self.mean()
        out = "Particles(r={!r}, v={!r}, t={!r}".format(pmean[0:3], pmean[3:6], self.epoch)
        out += ")"
        return out

    def reset_to_pseudo_prior(self):
        self.particles = self.initial_particles.copy()
        self.ln_wts = self.initial_ln_wts.copy()

    @property
    def epoch(self):
        """ Get the epoch time at which the particles (i.e., position and velocity) are defined
        """
        return self.rvprobability.epoch

    @property
    def orbits(self):
        """ Get a list of orbits corresponding to each stored particle
        """
        if self._orbits is None:
            self._orbits = [_Orbit(p[0:3], p[3:6], t=self.epoch) for p in self.particles]
        return self._orbits

    @property
    def lnpriors(self):
        """ Get an array of log prior probabilities for each particle
        """
        if self._lnpriors is None:
            self._lnpriors = np.array([self.rvprobability.lnprior(orbit) for orbit in self.orbits])
        return self._lnpriors

    def move(self, epoch):
        """ Move particles to the given epoch

        Parameters
        ----------
        epoch : astropy.time.Time

        Returns
        -------
        particles : array (num_particles, 6)
            Position and velocity for each particle at the new epoch
        """
        r, v = rv(self.orbits, epoch)
        return np.column_stack((r, v))

    def lnlike(self, orbits):
        """ Evaluate this epoch likelihood given one or more particles

        Parameters
        ----------
        orbits : List of Orbit class instances
            A list of the Orbit models for each particle, e.g., as generated by Particles.orbits

        Returns
        -------
        lnlike : array (num_particles, )
            Array of log-likelihood values for each particle
        """
        return np.array([self.rvprobability.lnlike(orbit) for orbit in orbits])

    def reweight(self, epoch_particles):
        """ Reweight particles using cross-epoch likelihoods

        The weights for the kth particle are,
            w_k = log(Prod_i L_(d_i | theta_k) / L(d_j | theta_k))
                = Sum_{i /= j} log(L(d_i | theta_k))
        where j is the current epoch, theta are the particle parameters, and d_i is the data from
        epoch i.

        Parameters
        ----------
        epoch_particles : Particles
            The particles object from another observation

        Returns
        -------
        status : bool
            Status (True means success). Side effect is to update the internal particles state.
        """
        status = True

        # Update particle weights for the input epoch given this epoch likelihood
        epoch_ln_wts = self.lnlike(epoch_particles.orbits) + epoch_particles.lnpriors
        epoch_ln_wts += epoch_particles.ln_wts

        # Update particle weights for this epoch given input epoch likelihood
        ln_wts = epoch_particles.lnlike(self.orbits) + self.lnpriors
        ln_wts += self.ln_wts

        # Append particles and weights in anticipation of resampling / downsampling
        self.particles = np.vstack((self.particles, epoch_particles.move(self.epoch)))
        self.ln_wts = np.append(ln_wts, epoch_ln_wts)

        if np.logaddexp.reduce(self.ln_wts) < -100.:
            status = False
        return status

    def resample(self, num_particles):
        """ Resample particles to achieve near-uniform weighting per particle

        Parameters
        ----------
        num_particles : int
            The number of particles to keep

        Returns
        -------
            None. Side effect is to update the internal particles state.
        """
        # TODO: Decide if resampling is really required here?
        self.particles, wts = utils.resample(self.particles,
                                             self.ln_wts, pod=True)
        self.ln_wts = np.log(wts)

        # Reset the stored log priors array
        self._lnpriors = None
        # Reset the stored orbits array
        self._orbits = None

        # Down-sample (without replacement) from the current particle list if
        # the requested number of particles does not match the current number.
        if self.particles.shape[0] > self.num_particles:
            ndx = np.random.choice(self.particles.shape[0], size=num_particles, replace=False)
            self.particles = self.particles[ndx, ]
            self.ln_wts = self.ln_wts[ndx]
        elif self.num_particles > self.particles.shape[0]:
            raise ValueError("Requested more particles than we have")
        return None

    def fuse(self, epoch_particles, verbose=False):
        """ Add particles from a new epoch and fuse all epoch particles and weights accordingly

        Parameters
        ----------
        epoch_particles : Particles
            The particles object from another observation
        verbose : bool, optional
            Enable verbose output to stdout

        Returns
        -------
            None. Side effect is to fuse the internal particles state.
        """
        num_part_out = self.num_particles
        # Update the weight values by 'cross-pollinating' epoch likelihoods
        status = self.reweight(epoch_particles)
        # Check that we have at least some non-zero weights for particles
        if status > 0 and verbose:
            print("All weights are negligible in Particles class")
            print("\tlog-norm:",np.logaddexp.reduce(self.ln_wts))
        # Resample the weights to maintain only significant particles
        self.resample(num_particles=num_part_out)
        return None

    def draw_orbit(self):
        """ Draw a single orbit model from the internal particle collection with probability
        proportional to the internal weights.

        Returns
        -------
        particle : array (1, 6)
            Position and velocity for one particle
        """
        prob = utils.get_normed_weights(self.ln_wts)
        particle_ndx = np.random.choice(self.num_particles, size=1, p=prob)
        p = self.particles[particle_ndx, :][0]
        return [_Orbit(p[0:3], p[3:6], t=self.epoch)]

    def mean(self):
        """ Evaluate the weighted mean of all particles

        Returns
        -------
        mean : (6,) array_like
            Weighted mean of the particle values (3*m, 3*m/s)
        """
        return np.average(self.particles, axis=0, weights=np.exp(self.ln_wts))
