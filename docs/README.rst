===================
SSAPy Documentation
===================

Building the docs
-----------------

First, build and *install* ``ssapy`` by following the
`Installing SSAPy <https://LLNL.github.io/SSAPy/installation.html>`_
section of the documentation.

If needed, install the documentation dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Then, from the ``docs`` directory, build the HTML documentation locally with:

.. code-block:: bash

    make html

You may need to run this command twice to generate all files correctly.

Then open ``_build/html/index.html`` in a browser to view the documentation locally.

Alternatively, you can run:

.. code-block:: bash

    sphinx-autobuild source _build/html

This starts a local server that watches for changes in ``docs/`` and
automatically rebuilds the HTML documentation while serving it at
http://127.0.0.1:8000/.

Note that if you update docstrings, you may need to rebuild and reinstall
``ssapy`` before rebuilding the documentation.