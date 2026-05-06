.. Copyright 2013-2024 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

.. _contribution-guide:

==================
Contribution Guide
==================

This guide is intended for developers and administrators who want to
contribute a new feature or bug fix to SSAPy. It assumes some familiarity
with Git and GitHub.

A pull request (PR) is the primary mechanism for contributing changes to
SSAPy. In general, each PR should correspond to a single completed feature,
bug fix, refactor, or documentation improvement. Smaller, focused PRs are
much easier to review and maintain than large PRs covering unrelated changes.

If possible, follow a **one PR, one feature** rule.

--------
Branches
--------

SSAPy's ``main`` branch contains the latest development work. Nearly all pull
requests should start from ``main`` and target ``main``.

There may also be release branches for major release series. Important bug
fixes may be backported to those branches by the maintainers as needed.

----------------------
Continuous Integration
----------------------

SSAPy uses `GitHub Actions <https://docs.github.com/en/actions>`_ for
continuous integration (CI). Every pull request and relevant push will trigger
automated checks such as package builds, tests, and documentation builds.

**Your PR will not be accepted until the required CI checks pass.**

Sometimes CI fails for reasons unrelated to your PR, such as transient network
issues or package manager failures. If this happens, inspect the failing job in
GitHub Actions before assuming your changes are at fault. If the failure appears
transient, rerun the workflow if you have permission, or push a small follow-up
commit / reopen the PR to trigger CI again.

The exact test matrix may change over time. Refer to the repository's GitHub
Actions workflow files for the current platforms and Python versions being
tested.

^^^^^^^^^^
Unit Tests
^^^^^^^^^^

SSAPy uses `pytest <https://pytest.org/>`_ for unit testing.

To run the full test suite locally:

.. code-block:: console

   $ pytest -v

To run a specific test file:

.. code-block:: console

   $ pytest -v tests/test_compute.py

To run a specific test:

.. code-block:: console

   $ pytest -v tests/test_compute.py::test_example

To see test output live while tests run:

.. code-block:: console

   $ pytest -v -s

If your PR changes core SSAPy functionality, please run the relevant tests
locally before submitting the PR. If you add new functionality or fix a bug,
please also add or update tests where appropriate.

^^^^^^^^^^^
Style Tests
^^^^^^^^^^^

SSAPy uses `Flake8 <https://flake8.pycqa.org/>`_ for style checking.

To run style checks locally:

.. code-block:: console

   $ flake8 ssapy tests devel

Please ensure that any code you contribute passes the project's style checks.

If additional static analysis tools are enabled in CI in the future, contributors
should also ensure those checks pass before requesting review.

^^^^^^^^^^^^^^^^^^^
Documentation Tests
^^^^^^^^^^^^^^^^^^^

SSAPy uses `Sphinx <https://www.sphinx-doc.org/>`_ to build documentation.

To build the documentation locally:

.. code-block:: console

   $ pip install -r docs/requirements.txt
   $ cd docs
   $ make clean
   $ make html

If your PR changes documentation, public APIs, or import structure, please build
the docs locally to make sure they still render correctly.

Depending on the current documentation configuration, additional system packages
such as ``graphviz`` may be required.

--------
Coverage
--------

SSAPy uses `Codecov <https://codecov.io/>`_ to report unit test coverage.

Coverage checks help identify whether changed code is exercised by tests.
Although coverage alone does not guarantee correctness, changes to core library
code should ideally be accompanied by tests covering the modified behavior.

-------------
Git Workflows
-------------

SSAPy development is active, and ``main`` may change frequently. To keep your
work manageable and reviewable, it is best to make changes on a dedicated branch.

^^^^^^^^^
Branching
^^^^^^^^^

Start by updating your local ``main`` branch and creating a new feature branch:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git checkout -b <descriptive_branch_name>

Make your changes, then commit them in logically grouped commits:

.. code-block:: console

   $ git add <files>
   $ git commit -m "<descriptive message>"

Push the branch to your fork:

.. code-block:: console

   $ git push origin <descriptive_branch_name> --set-upstream

Then open a pull request targeting ``main``.

Please use descriptive commit messages so future contributors can understand the
intent of the changes.

^^^^^^^^^^^^^^
Cherry-Picking
^^^^^^^^^^^^^^

If you already made commits on another branch and later decide to submit some of
them upstream, you can create a new branch from ``main`` and cherry-pick the
relevant commits.

First, find the commit hashes you want:

.. code-block:: console

   $ git log

Then create a branch from updated ``main`` and cherry-pick the commits:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git checkout -b <descriptive_branch_name>
   $ git cherry-pick <hash>

Push the new branch and open a pull request:

.. code-block:: console

   $ git push origin <descriptive_branch_name> --set-upstream

^^^^^^^^
Rebasing
^^^^^^^^

If ``main`` has moved forward and your branch has conflicts or is out of date,
rebase your branch on top of the latest ``main``:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git checkout <descriptive_branch_name>
   $ git rebase main

If Git reports conflicts, resolve them, then continue:

.. code-block:: console

   $ git add <resolved_files>
   $ git rebase --continue

Repeat until the rebase completes. Then force-push the updated branch:

.. code-block:: console

   $ git push --force origin <descriptive_branch_name>

^^^^^^^^^^^^^^^^^^^^^^^^^
Rebasing with cherry-pick
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also rebase manually by resetting a branch to ``main`` and cherry-picking
back the commits you want to keep.

Create a backup branch first:

.. code-block:: console

   $ git checkout <descriptive_branch_name>
   $ git branch tmp

Record the commit hashes you want to preserve:

.. code-block:: console

   $ git log

Update ``main`` and reset your working branch:

.. code-block:: console

   $ git checkout main
   $ git pull upstream main
   $ git checkout <descriptive_branch_name>
   $ git reset --hard main

Then cherry-pick the desired commits:

.. code-block:: console

   $ git cherry-pick <hash1>
   $ git cherry-pick <hash2>

Push the updated branch:

.. code-block:: console

   $ git push --force origin <descriptive_branch_name>

If everything looks good, remove the backup branch:

.. code-block:: console

   $ git branch --delete --force tmp

^^^^^^^^^^^^^^^^^^
Re-writing History
^^^^^^^^^^^^^^^^^^

Sometimes a branch diverges so far from ``main`` that a clean rebase is difficult.
If only the final result matters, you may choose to rewrite the branch history.

On the branch in question:

.. code-block:: console

   $ git merge main
   $ git reset main

At this point, your branch points to the same commit as ``main``, but your local
file modifications remain in the working tree. Review them with:

.. code-block:: console

   $ git status
   $ git diff

Then create a new, clean set of commits:

.. code-block:: console

   $ git add <files>
   $ git commit -m "<descriptive message>"

Finally, push the rewritten branch to your fork and open or update the PR:

.. code-block:: console

   $ git push --force origin <descriptive_branch_name>