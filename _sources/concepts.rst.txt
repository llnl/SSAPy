SSAPy Concepts
==============

This page explains concepts and terminology used within the SSAPy package.
It is intended to provide a high-level overview, while details can be found in other sections of the documentation.

.. _def-coordinate-systems:

Coordinate Systems
------------------
A `coordinate system <https://en.wikipedia.org/wiki/Coordinate_system>`_ is a set of one or more values which define a unique position in space. The SSAPy package is capable of using a variety of coordinate systems:

    - `Cartesian coordinates <https://en.wikipedia.org/wiki/Cartesian_coordinate_system>`_
    - Right ascension/Declination (``ra``/``dec``) of stars from catalog positions (J2000/ICRS)
    - Apparent positions
    - NTW

        - ``T`` gives the projection of ``rcoord - r`` along ``V``  (tangent to track)
        - ``W`` gives the projection of ``rcoord - r`` along ``(V cross r)`` (normal to plane)
        - ``N`` gives the projection of ``rcoord - r`` along ``(V cross (V cross r))`` (in plane, perpendicular to ``T``)

    - theta-phi-like coordinates: the system where ``theta`` is the angle between zenith and the point in question, and ``phi`` is the corresponding azimuthal angle. (radians)
    - lb-like coordinates & proper motions

        - If converting from theta-phi, sets ``b = pi - theta`` and renames ``phi`` -> ``l``.  Everything is in radians.

    - Orthographic tangent plane (radiants)

        - If converting from lb-like coordinates, the tangent plane is always chosen so that ``+Y`` is towards ``b = 90``, and ``+X`` is towards ``+l``.

    - Geocentric Celestial Reference Frame (GCRF)
    - International Terrestrial Reference System (ITRS)
    - Geocentric Celestial Reference System (GCRS) Cartesian coordinates
    - True Equator Mean Equinox frame (TEME) Cartesian coordinates

    Not every function will be able to make use of every coordinate system, so please be sure to read the documentation associated with a given operation.

.. _def-time-standards:

Time Standards
--------------

    - UTC vs others

.. _def-orbits:

Types of Standard Orbits
------------------------
While SSAPy is not technically limited to modeling a specific orbit, there are certain types of orbits which are more closely related to SSAPy's capabilities. Those orbit types are listed below:

- Low Earth Orbit (LEO)
- Mid-earth orbit (MEO)
- Geosynchronous Earth Orbit (GEO)
- Geostationary Earth Orbit
- Highly Elliptical Orbit (HEO)
- Cislunar Orbits

Of course, many other types of near Earth orbits are possible (i.e. lunar rectilinear halo orbits). The sky, or the universe as it may be, is the limit!

.. _def-models:

Types of Models
---------------
Accurately predicting the path of a spacecraft requires accounting for not only gravity, but also subtle influences like radiation pressure and atmospheric drag. SSAPy provides different models that can be combined depending on needs.

Gravitational models
^^^^^^^^^^^^^^^^^^^^
Several graviational models are provided to balance accuracy and performance.

A point source model assumes all the mass of the celestial body is concentrated at a single point in space, typically its center.  This is a simplification, but useful for long distances or preliminary calculations.
Using a point source model for the Sun or a planet to calculate the gravitational influence on a spacecraft is sufficient for initial orbit design or basic mission planning, but as the spacecraft gets closer, the limitations of the point source become apparent.

Harmonics describe the variations in the gravitational field across the sphere.
A model with more harmonics turned on provides a more accurate picture of the gravity field, but also requires more complex calculations.

For Earth and the Moon, more sophisticated models are needed due to their non-uniform mass distribution. These models incorporate:

- **Spherical Harmonics:** As mentioned above, with higher orders capturing features like mountains and valleys that affect the gravity field.
- **Mascons:** These are specific regions of high or low mass density that significantly impact the gravitational field.

These complexities make Earth/Lunar modeling more computationally demanding compared to a simple point source, but significantly improve the accuracy of orbit predictions.

Radiation pressure
^^^^^^^^^^^^^^^^^^
Radiation pressure is the force exerted by light or electromagnetic radiation on a surface. In space, where there's minimal drag from other particles, radiation pressure from the Sun can be a significant influence on a spacecraft's trajectory.

Radiation pressure depends on three key factors:

1. **Intensity of Sunlight:** Stronger sunlight exerts a greater force.
2. **Surface Properties:** Reflective surfaces experience a larger force compared to absorbent ones.
3. **Spacecraft Orientation:** The angle at which sunlight hits the spacecraft affects the force direction.

There are different levels of complexity in modeling radiation pressure, depending on the desired accuracy and computational resources:

- **Cannonball Model:** This is the simplest approach, treating the spacecraft as a perfect sphere.
  It calculates the force based on the spacecraft's area-to-mass ratio and reflectivity properties, assuming a constant force along the Sun-spacecraft direction.
  It's quick and easy to use, but lacks accuracy for complex spacecraft shapes.

- **N-Plate Model:** This model approximates the spacecraft using multiple flat plates, each with its own size, orientation, and reflectivity.
  It offers more detail than the cannonball model, capturing the effect of different surfaces on the spacecraft.
  The number of plates determines the accuracy, but also increases computational complexity.

- **Ray-Tracing Techniques:** This advanced method uses software that simulates the path of sunlight rays reflecting off the spacecraft's actual 3D geometry.
  It provides the most accurate picture of radiation pressure but requires significant computational power.

The choice of model depends on the specific mission requirements. For initial planning, a cannonball model might suffice. However, high-precision orbit determination for critical missions might require ray-tracing techniques.

Atmospheric Modeling
^^^^^^^^^^^^^^^^^^^^
Atmospheric drag is the resistance a spacecraft experiences due to collisions with gas molecules in a planet's atmosphere. While negligible at high altitudes, it becomes a significant force during atmospheric entry or when operating in low-Earth orbit.

Accurate atmospheric models are crucial for:

- Predicting spacecraft re-entry paths to ensure safe landing zones.
- Maintaining orbit stability for low-Earth satellites by compensating for drag-induced orbital decay.
- Optimizing fuel usage by accounting for drag during maneuvers.

.. _def-numerical-integrators:

Numerical Integrators
---------------------
Numerical integrators solve the complex differential equations that govern the motion of a spacecraft under various gravitational and environmental influences.
Different integrators offer trade-offs between accuracy, efficiency, and stability.

The step size in a numerical integrator defines the time interval between calculations. Smaller step sizes lead to more accurate results but require more computations. Choosing the right step size involves balancing accuracy needs with computational resources.
Different integrators handle step sizes differently:

- **Fixed-Step Integrators** use a constant time step size for calculations. They are simple to implement but can be inefficient, especially for rapidly changing forces or highly elliptical orbits.
- **Variable-Step Integrators** adjust the time step size dynamically based on the complexity of the motion. They are more efficient for problems with varying forces but can be more complex to implement.
- **Multi-Step Integrators** utilize information from previous time steps to improve accuracy. They can be efficient but may introduce stability issues for certain types of orbits.

The choice of integrator depends on several factors:

- **Orbit Type:** Highly elliptical orbits require more sophisticated methods than circular ones.
- **Force Model Complexity:** Simpler models might allow for simpler integrators, while complex models may necessitate more advanced methods.
- **Propagation Time:** Longer propagations benefit from efficient integrators.
- **Desired Accuracy:** Higher accuracy often comes at the cost of increased computation time.

.. _def-computing-considerations:

Computing Considerations
------------------------
..
    - Many orbits
    - vectorization
    - orbit sampling and propagation for error estimation (rvsampling)
        - Linking observations together
When propagating multiple orbits, make sure to leverage SSAPy's vectorization.
For instance, the :class:`.Orbit` class can represents either a single scalar orbit or a vector of orbits.

.. _def-other-codes:

Other Codes
-----------
Below is a list of other orbit propagation codes, both commercial and free. While these other pieces of software may have some features in common with SSAPy, we believe SSAPy brings a more complete list of capabilities within one package.

- `General Mission Analysis Tool (GMAT) <https://software.nasa.gov/software/GSC-17177-1>`_
- `Ansys Systems Tool Kit (STK) <https://www.ansys.com/products/missions/ansys-stk>`_
- `a.i. solutions FreeFlyer Astrodynamics Software <https://ai-solutions.com/freeflyer-astrodynamic-software/>`_
- `MathWorks MATLAB <https://www.mathworks.com/products/matlab.html?s_tid=hp_products_matlab>`_
- `AstroPy <https://docs.astropy.org/en/stable/index.html>`_
- `REBOUND <https://rebound.readthedocs.io/en/latest/>`_
- `REBOUNDx <https://reboundx.readthedocs.io/en/latest/index.html>`_
- `Nyx Space <https://nyxspace.com/nyxspace/showcase/>`_
