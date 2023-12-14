"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x):
    ":math:`f(x) = x`"
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x):
    ":math:`f(x) = -x`"
    return -x


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    return x if x > y else y


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    return abs(x - y) < 1e-2


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return float(math.log(x + EPS))


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    return d / (x + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1 / x


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    return -d / (x ** 2)


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    return d if x > 0 else 0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """

    def function(domain):
        """
        Args:
            domain: list of values

        Returns:
            range: list of values where each is applied with 'fn'
        """
        range = [fn(element) for element in domain]
        return range

    return function


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    """
    Args:
        ls: list of numbers
    Returns:
        list of negated numbers from ls
    """

    neg_map = map(neg)
    return neg_map(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def function(ls1, ls2):
        """
        Args:
            ls1: list of values
            ls2: list of values

        Returns:
            zip: a new list by applying fn(x, y) on each pair of elements.
        """
        zip = []
        for i in range(len(ls1)):
            zip.append(fn(ls1[i], ls2[i]))
        return zip

    return function


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    """
    Args:
        ls1: list of values
        ls2: list of values
    Returns:
        lists of values added pair-wise from two list
    """
    function = zipWith(add)
    return function(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """

    def function(ls):
        out = start
        for i in range(len(ls)):
            out = fn(ls[i], out)
        return out

    return function


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    """
    Args:
        ls: list of values
    Returns:
        sum of the list
    """

    function = reduce(add, 0)
    return function(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    """
    Args:
        ls: list of values
    Returns:
        product of the list
    """

    function = reduce(mul, 1)
    return function(ls)
