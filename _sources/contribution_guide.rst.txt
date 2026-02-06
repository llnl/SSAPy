.. Copyright 2013-2024 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

.. _contribution-guide:

==================
Contribution Guide
==================

This guide is intended for developers or administrators who want to
contribute a new feature or bugfix to SSAPy.
It assumes that you have at least some familiarity with Git and GitHub.
The guide will show a few examples of contributing workflows and discuss
the granularity of pull-requests (PRs). It will also discuss the tests your
PR must pass in order to be accepted into SSAPy.

First, what is a PR? Quoting `Bitbucket's tutorials <https://www.atlassian.com/git/tutorials/making-a-pull-request/>`_:

  Pull requests are a mechanism for a developer to notify team members that
  they have **completed a feature**. The pull request is more than just a
  notification—it’s a dedicated forum for discussing the proposed feature.

The emphasis is on a **completed feature**. The changes one proposes in a PR should
correspond to one feature/bugfix/extension/etc. One can create PRs with
changes relevant to different ideas, however reviewing such PRs becomes tedious
and error prone. If possible, try to follow the **one-PR-one-feature** rule.

--------
Branches
--------

SSAPy's ``main`` branch has the latest contributions. Nearly all pull
requests should start from ``main`` and target ``main``.

There is a branch for each major release series. Release branches
originate from ``main`` and have tags for each point release in the
series. For example, ``releases/v0.14`` might have tags for ``0.14.0``,
``0.14.1``, ``0.14.2``, etc. versions of SSAPy. We backport important bug
fixes to these branches, but we do not advance the package versions or
make other changes that would change the way SSAPy concretizes
dependencies. Currently, the maintainers manage these branches by
cherry-picking from ``main``. See :ref:`releases` for more
information.

----------------------
Continuous Integration
----------------------

SSAPy uses `Github Actions <https://docs.github.com/en/actions>`_ for Continuous Integration
testing. This means that every time you submit a pull request, a series of tests will
be run to make sure you didn't accidentally introduce any bugs into SSAPy. **Your PR
will not be accepted until it passes all of these tests.** While you can certainly wait
for the results of these tests after submitting a PR, we recommend that you run them
locally to speed up the review process.

.. note::

   Oftentimes, CI will fail for reasons other than a problem with your PR.
   For example, apt-get, pip, or homebrew will fail to download one of the
   dependencies for the test suite, or a transient bug will cause the unit tests
   to timeout. If any job fails, click the "Details" link and click on the test(s)
   that is failing. If it doesn't look like it is failing for reasons related to
   your PR, you have two options. If you have write permissions for the SSAPy
   repository, you should see a "Restart workflow" button on the right-hand side. If
   not, you can close and reopen your PR to rerun all of the tests. If the same
   test keeps failing, there may be a problem with your PR. If you notice that
   every recent PR is failing with the same error message, it may be that an issue
   occurred with the CI infrastructure or one of SSAPy's dependencies put out a
   new release that is causing problems. If this is the case, please file an issue.


We currently test against Python 3.10 on both macOS and Linux and
perform 3 types of tests:

.. _cmd-ssapy-unit-test:

^^^^^^^^^^
Unit Tests
^^^^^^^^^^

Unit tests ensure that core SSAPy features like fetching or spec resolution are
working as expected. If your PR only adds new packages or modifies existing ones,
there's very little chance that your changes could cause the unit tests to fail.
However, if you make changes to SSAPy's core libraries, you should run the unit
tests to make sure you didn't break anything.

Since they test things like fetching from VCS repos, the unit tests require
`git <https://git-scm.com/>`_, `mercurial <https://www.mercurial-scm.org/>`_,
and `subversion <https://subversion.apache.org/>`_ to run. Make sure these are
installed on your system and can be found in your ``PATH``. All of these can be
installed with SSAPy or with your system package manager.

To run *all* of the unit tests, use:

.. code-block:: console

   $ ssapy unit-test

These tests may take several minutes to complete. If you know you are
only modifying a single SSAPy feature, you can run subsets of tests at a
time.  For example, this would run all the tests in
``tests/*.py``:

.. code-block:: console

   $ ssapy unit-test tests/*.py

This allows you to develop iteratively: make a change, test that change,
make another change, test that change, etc.  We use `pytest
<http://pytest.org/>`_ as our tests framework, and these types of
arguments are just passed to the ``pytest`` command underneath. See `the
pytest docs
<https://doc.pytest.org/en/latest/how-to/usage.html#specifying-which-tests-to-run>`_
for more details on test selection syntax.

By default, ``pytest`` captures the output of all unit tests, and it will
print any captured output for failed tests. Sometimes it's helpful to see
your output interactively, while the tests run (e.g., if you add print
statements to a unit tests).  To see the output *live*, use the ``-s``
argument to ``pytest``:

.. code-block:: console

   $ ssapy unit-test -s --list-long tests/architecture.py::test_platform

Unit tests are crucial to making sure bugs aren't introduced into
SSAPy. If you are modifying core SSAPy libraries or adding new
functionality, please add new unit tests for your feature, and consider
strengthening existing tests.  You will likely be asked to do this if you
submit a pull request to the SSAPy project on GitHub.  Check out the
`pytest docs <http://pytest.org/>`_ and feel free to ask for guidance on
how to write tests!

^^^^^^^^^^^^
Style Tests
^^^^^^^^^^^^

SSAPy uses `Flake8 <http://flake8.pycqa.org/en/latest/>`_ to test for
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ conformance and
`mypy <https://mypy.readthedocs.io/en/stable/>` for type checking. PEP 8 is
a series of style guides for Python that provide suggestions for everything
from variable naming to indentation. In order to limit the number of PRs that
were mostly style changes, we decided to enforce PEP 8 conformance. Your PR
needs to comply with PEP 8 in order to be accepted, and if it modifies the
ssapy library it needs to successfully type-check with mypy as well.

Testing for compliance with ssapy's style is easy. Simply run the ``ssapy style``
command:

.. code-block:: console

   $ ssapy style

``ssapy style`` has a couple advantages over running the tools by hand:

#. It only tests files that you have modified since branching off of
   ``main``.

#. It works regardless of what directory you are in.

#. It automatically adds approved exemptions from the ``flake8``
   checks. For example, URLs are often longer than 80 characters, so we
   exempt them from line length checks. We also exempt lines that start
   with "homepage", "url", "version", "variant", "depends_on", and
   "extends" in ``package.py`` files.  This is now also possible when directly
   running flake8 if you can use the ``ssapy`` formatter plugin included with
   ssapy.

More approved flake8 exemptions can be found
`here <https://github.com/LLNL/SSAPy/blob/main/.flake8>`_.

If all is well, you'll see something like this:

.. code-block:: console

   $ run-flake8-tests
   Dependencies found.
   =======================================================
   flake8: running flake8 code checks on SSAPy.

   Modified files:

     var/ssapy/repos/builtin/packages/hdf5/package.py
     var/ssapy/repos/builtin/packages/hdf/package.py
     var/ssapy/repos/builtin/packages/netcdf/package.py
   =======================================================
   Flake8 checks were clean.

However, if you aren't compliant with PEP 8, flake8 will complain:

.. code-block:: console

   var/ssapy/repos/builtin/packages/netcdf/package.py:26: [F401] 'os' imported but unused
   var/ssapy/repos/builtin/packages/netcdf/package.py:61: [E303] too many blank lines (2)
   var/ssapy/repos/builtin/packages/netcdf/package.py:106: [E501] line too long (92 > 79 characters)
   Flake8 found errors.

Most of the error messages are straightforward, but if you don't understand what
they mean, just ask questions about them when you submit your PR. The line numbers
will change if you add or delete lines, so simply run ``ssapy style`` again
to update them.

.. tip::

   Try fixing flake8 errors in reverse order. This eliminates the need for
   multiple runs of ``ssapy style`` just to re-compute line numbers and
   makes it much easier to fix errors directly off of the CI output.


^^^^^^^^^^^^^^^^^^^
Documentation Tests
^^^^^^^^^^^^^^^^^^^

SSAPy uses `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ to build its
documentation. In order to prevent things like broken links and missing imports,
we added documentation tests that build the documentation and fail if there
are any warning or error messages.

Building the documentation requires several dependencies:

* docutils
* sphinx
* sphinx-rtd-theme
* sphinx-copybutton
* sphinx-autobuild
* sphinx-tabs
* sphinx-automodapi
* myst-parser
* graphviz

All of these can be installed with Pip, e.g.

.. code-block:: console

   $ python3 -m pip install docutils sphinx, sphinx-rtd-theme sphinx-copybutton sphinx-autobuild sphinx-tabs sphinx-automodapi myst-parser graphviz

Once all of the dependencies are installed, you can try building the documentation:

.. code-block:: console

   $ cd path/to/ssapy/docs/
   $ make clean
   $ make

If you see any warning or error messages, you will have to correct those before your PR
is accepted. If you are editing the documentation, you should be running the
documentation tests to make sure there are no errors. Documentation changes can result
in some obfuscated warning messages. If you don't understand what they mean, feel free
to ask when you submit your PR.

--------
Coverage
--------

SSAPy uses `Codecov <https://codecov.io/>`_ to generate and report unit test
coverage. This helps us tell what percentage of lines of code in SSAPy are
covered by unit tests. Although code covered by unit tests can still contain
bugs, it is much less error prone than code that is not covered by unit tests.

Codecov provides `browser extensions <https://github.com/codecov/sourcegraph-codecov>`_
for Google Chrome and Firefox. These extensions integrate with GitHub
and allow you to see coverage line-by-line when viewing the SSAPy repository.
If you are new to SSAPy, a great way to get started is to write unit tests to
increase coverage!

Unlike with CI on Github Actions Codecov tests are not required to pass in order for your
PR to be merged. If you modify core SSAPy libraries, we would greatly
appreciate unit tests that cover these changed lines. Otherwise, we have no
way of knowing whether or not your changes introduce a bug. If you make
substantial changes to the core, we may request unit tests to increase coverage.

.. note::

   You may notice that the Codecov tests fail even though
   you didn't modify any core files. This means that SSAPy's overall coverage
   has increased since you branched off of ``main``. This is a good thing!
   If you really want to get the Codecov tests to pass, you can rebase off of
   the latest ``main``, but again, this is not required.


-------------
Git Workflows
-------------

SSAPy is still in the beta stages of development. Most of our users run off of
the ``main`` branch, and fixes and new features are constantly being merged. So
how do you keep up-to-date with upstream while maintaining your own local
differences and contributing PRs to SSAPy?

^^^^^^^^^
Branching
^^^^^^^^^

The easiest way to contribute a pull request is to make all of your changes on
new branches. Make sure your ``main`` is up-to-date and create a new branch
off of it:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git branch <descriptive_branch_name>
   $ git checkout <descriptive_branch_name>

Here we assume that the local ``main`` branch tracks the upstream ``main``
branch of SSAPy. This is not a requirement and you could also do the same with
remote branches. But for some it is more convenient to have a local branch that
tracks upstream.

Normally we prefer that commits pertaining to a package ``<package-name>`` have
a message ``<package-name>: descriptive message``. It is important to add
descriptive message so that others, who might be looking at your changes later
(in a year or maybe two), would understand the rationale behind them.

Now, you can make your changes while keeping the ``main`` branch pure.
Edit a few files and commit them by running:

.. code-block:: console

   $ git add <files_to_be_part_of_the_commit>
   $ git commit --message <descriptive_message_of_this_particular_commit>

Next, push it to your remote fork and create a PR:

.. code-block:: console

   $ git push origin <descriptive_branch_name> --set-upstream

GitHub provides a `tutorial <https://help.github.com/articles/about-pull-requests/>`_
on how to file a pull request. When you send the request, make ``main`` the
destination branch.

If you need this change immediately and don't have time to wait for your PR to
be merged, you can always work on this branch. But if you have multiple PRs,
another option is to maintain a Frankenstein branch that combines all of your
other branches:

.. code-block:: console

   $ git checkout main
   $ git branch <your_modified_main_branch>
   $ git checkout <your_modified_main_branch>
   $ git merge <descriptive_branch_name>

This can be done with each new PR you submit. Just make sure to keep this local
branch up-to-date with upstream ``main`` too.

^^^^^^^^^^^^^^
Cherry-Picking
^^^^^^^^^^^^^^

What if you made some changes to your local modified ``main`` branch and already
committed them, but later decided to contribute them to SSAPy? You can use
cherry-picking to create a new branch with only these commits.

First, check out your local modified ``main`` branch:

.. code-block:: console

   $ git checkout <your_modified_main_branch>

Now, get the hashes of the commits you want from the output of:

.. code-block:: console

   $ git log

Next, create a new branch off of upstream ``main`` and copy the commits
that you want in your PR:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git branch <descriptive_branch_name>
   $ git checkout <descriptive_branch_name>
   $ git cherry-pick <hash>
   $ git push origin <descriptive_branch_name> --set-upstream

Now you can create a PR from the web-interface of GitHub. The net result is as
follows:

#. You patched your local version of SSAPy and can use it further.
#. You "cherry-picked" these changes in a stand-alone branch and submitted it
   as a PR upstream.

Should you have several commits to contribute, you could follow the same
procedure by getting hashes of all of them and cherry-picking to the PR branch.

.. note::

   It is important that whenever you change something that might be of
   importance upstream, create a pull request as soon as possible. Do not wait
   for weeks/months to do this, because:

   #. you might forget why you modified certain files
   #. it could get difficult to isolate this change into a stand-alone clean PR.

^^^^^^^^
Rebasing
^^^^^^^^

Other developers are constantly making contributions to SSAPy, possibly on the
same files that your PR changed. If their PR is merged before yours, it can
create a merge conflict. This means that your PR can no longer be automatically
merged without a chance of breaking your changes. In this case, you will be
asked to rebase on top of the latest upstream ``main``.

First, make sure your ``main`` branch is up-to-date:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main

Now, we need to switch to the branch you submitted for your PR and rebase it
on top of ``main``:

.. code-block:: console

   $ git checkout <descriptive_branch_name>
   $ git rebase main

Git will likely ask you to resolve conflicts. Edit the file that it says can't
be merged automatically and resolve the conflict. Then, run:

.. code-block:: console

   $ git add <file_that_could_not_be_merged>
   $ git rebase --continue

You may have to repeat this process multiple times until all conflicts are resolved.
Once this is done, simply force push your rebased branch to your remote fork:

.. code-block:: console

   $ git push --force origin <descriptive_branch_name>

^^^^^^^^^^^^^^^^^^^^^^^^^
Rebasing with cherry-pick
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also perform a rebase using ``cherry-pick``. First, create a temporary
backup branch:

.. code-block:: console

   $ git checkout <descriptive_branch_name>
   $ git branch tmp

If anything goes wrong, you can always go back to your ``tmp`` branch.
Now, look at the logs and save the hashes of any commits you would like to keep:

.. code-block:: console

   $ git log

Next, go back to the original branch and reset it to ``main``.
Before doing so, make sure that you local ``main`` branch is up-to-date
with upstream:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git checkout <descriptive_branch_name>
   $ git reset --hard main

Now you can cherry-pick relevant commits:

.. code-block:: console

   $ git cherry-pick <hash1>
   $ git cherry-pick <hash2>

Push the modified branch to your fork:

.. code-block:: console

   $ git push --force origin <descriptive_branch_name>

If everything looks good, delete the backup branch:

.. code-block:: console

   $ git branch --delete --force tmp

^^^^^^^^^^^^^^^^^^
Re-writing History
^^^^^^^^^^^^^^^^^^

Sometimes you may end up on a branch that has diverged so much from ``main``
that it cannot easily be rebased. If the current commits history is more of
an experimental nature and only the net result is important, you may rewrite
the history.

First, merge upstream ``main`` and reset you branch to it. On the branch
in question, run:

.. code-block:: console

   $ git merge main
   $ git reset main

At this point your branch will point to the same commit as ``main`` and
thereby the two are indistinguishable. However, all the files that were
previously modified will stay as such. In other words, you do not lose the
changes you made. Changes can be reviewed by looking at diffs:

.. code-block:: console

   $ git status
   $ git diff

The next step is to rewrite the history by adding files and creating commits:

.. code-block:: console

   $ git add <files_to_be_part_of_commit>
   $ git commit --message <descriptive_message>

After all changed files are committed, you can push the branch to your fork
and create a PR:

.. code-block:: console

   $ git push origin --set-upstream
