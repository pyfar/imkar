.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/pyfar/imkar/issues.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

imkarhttps://github.com/pyfar/pyfar could always use more documentation, whether as part of the
official imkar docs, in docstrings, or even on the web in blog posts,
articles, and such.

Get Started!
------------

Ready to contribute? Here's how to set up `imkar` for local development.

1. Fork the `imkar` repo on GitHub.
2. Clone your fork locally::

    $ git clone https://github.com/pyfar/imkar.git

3. Install your local copy into a virtual environment. Assuming you have Anaconda or Miniconda installed, this is how you set up your fork for local development::

    $ conda create --name imkar python pip
    $ conda activate imkar
    $ cd imkar
    $ pip install -r requirements_dev.txt
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 imkar tests
    $ pytest

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. Check https://app.circleci.com/pipelines/github/pyfar/imkar
   and make sure that the tests pass for all supported Python versions.


Testing Guidelines
-----------------------
imkar uses test-driven development based on `three steps <https://martinfowler.com/bliki/TestDrivenDevelopment.html>`_ and `continuous integration <https://en.wikipedia.org/wiki/Continuous_integration>`_ to test and monitor the code.
In the following, you'll find a guideline.

- The main tool used for testing is `pytest <https://docs.pytest.org/en/stable/index.html>`_.
- All tests are located in the *tests/* folder.
- Make sure that all important parts of imkar are covered by the tests. This can be checked using *coverage* (see below).
- In case of imkar, mainly **state verification** is applied in the tests. This means that the outcome of a function is compared to a desired value (``assert ...``). For more information, it is refered to `Martin Fowler's article <https://martinfowler.com/articles/mocksArentStubs.html.>`_.

Tips
~~~~~~~~~~~
Pytest provides several, sophisticated functionalities which could reduce the effort of implementing tests.

- Similar tests executing the same code with different variables can be `parametrized <https://docs.pytest.org/en/stable/example/parametrize.html>`_. An example is ``test___eq___differInPoints`` in *test_coordinates.py*.

- Run a single test with

    $ pytest tests/test_plot.py::test_line_plots

- Exclude tests (for example the time consuming test of plot) with

    $ pytest -k 'not plot'

- Create an html report on the test `coverage <https://coverage.readthedocs.io/en/coverage-5.5/>`_ with

    $ pytest --cov=. --cov-report=html

- Feel free to add more recommendations on useful pytest functionalities here. Consider, that a trade-off between easy implemention and good readability of the tests needs to be found.

Fixtures
~~~~~~~~
This section is not specific to imkar, but oftentimes refers to features and examples implemented in the pyfar package which is one of the main dependencies of `imkar <https://https://github.com/pyfar/pyfar>`_.

"Software test fixtures initialize test functions. They provide a fixed baseline so that tests execute reliably and produce consistent, repeatable, results. Initialization may setup services, state, or other operating environments. These are accessed by test functions through arguments; for each fixture used by a test function there is typically a parameter (named after the fixture) in the test function’s definition." (from https://docs.pytest.org/en/stable/fixture.html)

- All fixtures are implemented in *conftest.py*, which makes them automatically available to all tests. This prevents from implementing redundant, unreliable code in several test files.
- Typical fixtures are imkar objects with varying properties, stubs as well as functions need for initiliazing tests.
- Define the variables used in the tests only once, either in the test itself or in the definition of the fixture. This assures consistency and prevents from failing tests due to the definition of variables with the same purpose at different positions or in different files.

Have a look at already implemented fixtures in *confest.py*.

**Dummies**

If the objects used in the tests have arbitrary properties, tests are usually better to read, when these objects are initialized within the tests. If the initialization requires several operations or the object has non-arbitrary properties, this is a hint to use a fixture.
Good examples illustrating these two cases are the initializations in pyfar's *test_signal.py* vs. the sine and impulse signal fixtures in pyfar's *conftest.py*.

**Stubs**

Stubs mimic actual objects, but have minimum functionality and **fixed, well defined properties**. They are **only used in cases, when a dependence on the actual class is prohibited**.
This is the case, when functionalities of the class itself or methods it depends on are tested. Examples are the tests of the pyfar Signal class and its methods in *test_signal.py* and *test_fft.py*.

It requires a little more effort to implement stubs of classes. Therefore, stub utilities are provided in and imported in *confest.py*, where the actual stubs are implemented.

- Note: the stub utilities are not meant to be imported to test files directly or used for other purposes than testing. They solely provide functionality to create fixtures.
- The utilities simplify and harmonize testing within package and improve the readability and reliability.
- The implementation as the private submodule ``pyfar.testing.stub_utils``  further allows the use of similar stubs in related packages with pyfar dependency (e.g. other packages from the pyfar family).

**Mocks**

Mocks are similar to stubs but used for **behavioral verification**. For example, a mock can replace a function or an object to check if it is called with correct parameters. A main motivation for using mocks is to avoid complex or time-consuming external dependencies, for example database queries.

- A typical use case of mocks in the pyfar context is hardware communication, for example reading and writing of large files or audio in- and output. These use cases are rare compared to tests performing state verification.
- In contrast to some other guidelines on mocks, external dependencies do **not** need to be mocked in general. Failing tests due to changes in external packages are meaningful hints to modify the code.
- Examples of internal mocking can be found in pyfar's *test_io.py*, indicated by the pytest ``@patch`` calls.


Writing the Documentation
-------------------------

imkar follows the `numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for the docstring. A docstring has to consist at least of

- A short and/or extended summary,
- the Parameters section, and
- the Returns section

Optional fields that are often used are

- References,
- Examples, and
- Notes

Here are a few tips to make things run smoothly

- Use the tags ``:py:func:``, ``:py:mod:``, and ``:py:class:`` to reference pyfar functions, modules, and classes: For example ``:py:func:`~pyfar.plot.time``` for a link that displays only the function name. For links with custom text use ``:py:mod:`plot functions <pyfar.plot>```.
- Code snippets and values as well as external modules, classes, functions are marked by double ticks \`\` to appear in mono spaced font, e.g., ``x=3`` or ``pyfar.Signal``.
- Parameters, returns, and attributes are marked by single ticks \` to appear as emphasized text, e.g., *unit*.
- Use ``[#]_`` and ``.. [#]`` to get automatically numbered footnotes.
- Do not use footnotes in the short summary. Only use footnotes in the extended summary if there is a short summary. Otherwise, it messes with the auto-footnotes.
- If a method or class takes or returns pyfar objects for example write ``parameter_name : Signal``. This will create a link to the ``pyfar.Signal`` class.
- Plots can be included in by using the prefix ``.. plot::`` followed by an empty line and an indented block containing the code for the plot.

See the `Sphinx homepage <https://www.sphinx-doc.org>`_ for more information.

Building the Documentation
--------------------------

You can build the documentation of your branch using Sphinx by executing the make script inside the docs folder.

.. code-block:: console

    $ cd docs/
    $ make html

After Sphinx finishes you can open the generated html using any browser

.. code-block:: console

    $ docs/_build/index.html

Note that some warnings are only shown the first time you build the
documentation. To show the warnings again use

.. code-block:: console

    $ make clean

before building the documentation.



Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run:

..  code-block:: console

    $ bump2version patch # possible: major / minor / patch
    $ git push
    $ git push --tags

CircleCI will then deploy to PyPI if tests pass.

To manually build the package and upload to pypi run::

.. code-block:: console

    $ python setup.py sdist bdist_wheel
    $ twine upload dist/*
