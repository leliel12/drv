# DRV Processes (Decision with Reduction of Variability).


[![Build Status](https://travis-ci.org/leliel12/sc_drv.svg?branch=master)](https://travis-ci.org/leliel12/sc_drv)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/scikit-criteria)

DRV processes developed on top of [Scikit-Criteria](http://scikit-criteria.org).

DRV processes have been developed to support Group Decision
Making. They are applicable to the cases in which
all members of the group operate in the same organization and, therefore,
they must share organizational values, knowledge and preferences.
Assumes that it is necessary to generate agreement on the preferences of
group members.

## Support

-   **Contact:** jbc.develop@gmail.com nluczywo@gmail.com
-   **Issues:** If you have issues please report them as a issue
    here: https://github.com/leliel12/sc_drv/issues

## Instalation

The easiest way to install sc_drv is using `pip`

```bash
    $ pip install -U sc-drv
```


If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).
Building numpy and scipy from source can be complex (especially on Windows) and
requires careful configuration to ensure that they link against an optimized
implementation of linear algebra routines.
Instead, use a third-party distribution as described below.


### Third-party Distributions

If you don't already have a python installation with numpy and scipy, we
recommend to install either via your package manager or via a python bundle.
These come with numpy, scipy, matplotlib and many other helpful
scientific and data processing libraries.

Available options are:

#### Canopy and Anaconda for all supported platforms

[Canopy](https://www.enthought.com/products/canopy) and
[Anaconda](https://www.continuum.io/downloads) both ship a recent
version of Python, in addition to a large set of scientific python
library for Windows, Mac OSX and Linux.

## Documentation

- [Interactive Tutorial](https://mybinder.org/v2/gh/leliel12/sc_drv/master?filepath=notebooks%2Ftutorial.ipynb)


## Authors

-   Juan B. Cabral ([IATE-OAC-CONICET](https://iate.oac.uncor.edu),
    [FCEIA-UNR](https://web.fceia.unr.edu.ar/es/)) <jbc.develop@gmail.com>
-   Nadia A. Luczywo ([LIMI-FCEFyN-UNC](http://www.portal.efn.uncor.edu),
    [FCE-UNC](http://www.eco.unc.edu.ar/),
    [SECyT-UNC](https://www.unc.edu.ar/ciencia-y-tecnolog%C3%ADa/))
    <nluczywo@gmail.com>
-   José L. Zanazzi ([LIMI-FCEFyN-UNC](http://www.portal.efn.uncor.edu))
    <jl.zanazzi@gmail.com>


## References

-   Zanazzi, J. L., Gomes, L. F. A. M., & Dimitroff, M. (2014). Group decision
    making applied to preventive maintenance systems. Pesquisa Operacional,
    34(1), 91-105.
-   Cabral, J. B., Luczywo, N. A., & Zanazzi, J. L. (2016). Scikit-Criteria:
    colección de métodos de análisis multi-criterio integrado al stack
    científico de Python. In XIV Simposio Argentino de Investigación Operativa
    (SIO 2016)-JAIIO 45 (Tres de Febrero, 2016).
