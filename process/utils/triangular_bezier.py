'''
    We use the code in "https://github.com/dhermes/bezier/blob/main/src/python/bezier/triangle.py" here
'''
import numpy as np

def evaluate_barycentric(nodes, degree, lambda1, lambda2, lambda3):
    r"""Compute a point on a triangle.

    Evaluates :math:`B\left(\lambda_1, \lambda_2, \lambda_3\right)` for a
    B |eacute| zier triangle / triangle defined by ``nodes``.

    .. note::

       There is also a Fortran implementation of this function, which
       will be used if it can be built.

    Args:
        nodes (numpy.ndarray): Control point nodes that define the triangle.
        degree (int): The degree of the triangle define by ``nodes``.
        lambda1 (float): Parameter along the reference triangle.
        lambda2 (float): Parameter along the reference triangle.
        lambda3 (float): Parameter along the reference triangle.

    Returns:
        numpy.ndarray: The evaluated point as a ``D x 1`` array (where ``D``
        is the ambient dimension where ``nodes`` reside).
    """
    dimension, num_nodes = nodes.shape
    binom_val = 1.0
    result = np.zeros((dimension, 1), order="F")
    index = num_nodes - 1
    result[:, 0] += nodes[:, index]
    # curve evaluate_multi_barycentric() takes arrays.
    lambda1 = np.asfortranarray([lambda1])
    lambda2 = np.asfortranarray([lambda2])
    for k in range(degree - 1, -1, -1):
        # We want to go from (d C (k + 1)) to (d C k).
        binom_val = (binom_val * (k + 1)) / (degree - k)
        index -= 1  # Step to last element in column.
        #     k = d - 1, d - 2, ...
        # d - k =     1,     2, ...
        # We know column k has (d - k + 1) elements.
        new_index = index - degree + k  # First element in column.
        col_nodes = nodes[:, new_index : index + 1]  # noqa: E203
        col_nodes = np.asfortranarray(col_nodes)
        col_result = evaluate_multi_barycentric(
            col_nodes, lambda1, lambda2
        )
        result *= lambda3
        result += binom_val * col_result
        # Update index for next iteration.
        index = new_index
    return result

def evaluate_barycentric_multi(nodes, degree, param_vals, dimension):
    r"""Compute multiple points on the triangle.

    .. note::

       There is also a Fortran implementation of this function, which
       will be used if it can be built.

    Args:
        nodes (numpy.ndarray): Control point nodes that define the triangle.
        degree (int): The degree of the triangle define by ``nodes``.
        param_vals (numpy.ndarray): Array of parameter values (as a
            ``N x 3`` array).
        dimension (int): The dimension the triangle lives in.

    Returns:
        numpy.ndarray: The evaluated points, where columns correspond to
        rows of ``param_vals`` and the rows to the dimension of the
        underlying triangle.
    """
    num_vals, _ = param_vals.shape
    result = np.empty((dimension, num_vals), order="F")
    for index, (lambda1, lambda2, lambda3) in enumerate(param_vals):
        result[:, index] = evaluate_barycentric(
            nodes, degree, lambda1, lambda2, lambda3
        )[:, 0]
    return result

def evaluate_multi_barycentric(nodes, lambda1, lambda2):
    r"""Evaluates a B |eacute| zier type-function.

    Of the form

    .. math::

       B(\lambda_1, \lambda_2) = \sum_j \binom{n}{j}
           \lambda_1^{n - j} \lambda_2^j \cdot v_j

    for some set of vectors :math:`v_j` given by ``nodes``. This uses the
    more efficient :func:`.evaluate_multi_vs` until degree 55, at which point
    :math:`\binom{55}{26}` and other coefficients cannot be computed exactly.
    For degree 55 and higher, the classical de Casteljau algorithm will be
    used via :func:`.evaluate_multi_de_casteljau`.

    .. note::

       There is also a Fortran implementation of this function, which
       will be used if it can be built.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        lambda1 (numpy.ndarray): Parameters along the curve (as a
            1D array).
        lambda2 (numpy.ndarray): Parameters along the curve (as a
            1D array). Typically we have ``lambda1 + lambda2 == 1``.

    Returns:
        numpy.ndarray: The evaluated points as a two dimensional
        NumPy array, with the columns corresponding to each pair of parameter
        values and the rows to the dimension.
    """
    _, num_nodes = nodes.shape
    # NOTE: The computation of (degree C k) values in ``evaluate_multi_vs``
    #       starts to introduce round-off when computing (55 C 26). For very
    #       large degree, we ditch the VS algorithm and use de Casteljau
    #       (which has quadratic runtime and cubic space usage).
    if num_nodes > 55:
        return evaluate_multi_de_casteljau(nodes, lambda1, lambda2)

    return evaluate_multi_vs(nodes, lambda1, lambda2)

def evaluate_multi_de_casteljau(nodes, lambda1, lambda2):
    r"""Evaluates a B |eacute| zier type-function.

    Of the form

    .. math::

       B(\lambda_1, \lambda_2) = \sum_j \binom{n}{j}
           \lambda_1^{n - j} \lambda_2^j \cdot v_j

    for some set of vectors :math:`v_j` given by ``nodes``.

    Does so via the de Castljau algorithm:

    .. math::

       \begin{align*}
       v_j^{(n)} &= v_j \\
       v_j^{(k)} &= \lambda_1 \cdot v_j^{(k + 1)} +
           \lambda_2 \cdot v_{j + 1}^{(k + 1)} \\
       B(\lambda_1, \lambda_2) &= v_0^{(0)}
       \end{align*}

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        lambda1 (numpy.ndarray): Parameters along the curve (as a
            1D array).
        lambda2 (numpy.ndarray): Parameters along the curve (as a
            1D array). Typically we have ``lambda1 + lambda2 == 1``.

    Returns:
        numpy.ndarray: The evaluated points as a two dimensional
        NumPy array, with the columns corresponding to each pair of parameter
        values and the rows to the dimension.
    """
    # NOTE: We assume but don't check that lambda2 has the same shape.
    (num_vals,) = lambda1.shape
    dimension, num_nodes = nodes.shape
    degree = num_nodes - 1

    lambda1_wide = np.empty((dimension, num_vals, degree), order="F")
    lambda2_wide = np.empty((dimension, num_vals, degree), order="F")
    workspace = np.empty((dimension, num_vals, degree), order="F")
    for index in range(num_vals):
        lambda1_wide[:, index, :] = lambda1[index]
        lambda2_wide[:, index, :] = lambda2[index]
        workspace[:, index, :] = (
            lambda1[index] * nodes[:, :degree] + lambda2[index] * nodes[:, 1:]
        )

    for index in range(degree - 1, 0, -1):
        workspace[:, :, :index] = (
            lambda1_wide[:, :, :index] * workspace[:, :, :index]
            + lambda2_wide[:, :, :index]
            * workspace[:, :, 1 : (index + 1)]  # noqa: E203
        )

    # NOTE: This returns an array with `evaluated.flags.owndata` false, though
    #       it is Fortran contiguous.
    return workspace[:, :, 0]

def evaluate_multi_vs(nodes, lambda1, lambda2):
    r"""Evaluates a B |eacute| zier type-function.

    .. _VS Algorithm: https://doi.org/10.1016/0167-8396(86)90018-X

    Of the form

    .. math::

       B(\lambda_1, \lambda_2) = \sum_j \binom{n}{j}
           \lambda_1^{n - j} \lambda_2^j \cdot v_j

    for some set of vectors :math:`v_j` given by ``nodes``.

    Does so via a modified Horner's method (the `VS Algorithm`_) for each
    pair of values in ``lambda1`` and ``lambda2``.

    .. math::

       \begin{align*}
       w_0 &= \lambda_1 v_0 \\
       w_j &= \lambda_1 \left[w_{j - 1} +
           \binom{n}{j} \lambda_2^j v_j\right] \\
       w_n &= w_{n - 1} + \lambda_2^n v_n \\
       B(\lambda_1, \lambda_2) &= w_n
       \end{align*}

    Additionally, binomial coefficients are computed by utilizing the fact that
    :math:`\binom{n}{j} = \binom{n}{j - 1} \frac{n - j + 1}{j}`.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        lambda1 (numpy.ndarray): Parameters along the curve (as a
            1D array).
        lambda2 (numpy.ndarray): Parameters along the curve (as a
            1D array). Typically we have ``lambda1 + lambda2 == 1``.

    Returns:
        numpy.ndarray: The evaluated points as a two dimensional
        NumPy array, with the columns corresponding to each pair of parameter
        values and the rows to the dimension.
    """
    # NOTE: We assume but don't check that lambda2 has the same shape.
    (num_vals,) = lambda1.shape
    dimension, num_nodes = nodes.shape
    degree = num_nodes - 1
    # Resize as row vectors for broadcast multiplying with
    # columns of ``nodes``.
    lambda1 = lambda1[np.newaxis, :]
    lambda2 = lambda2[np.newaxis, :]
    result = np.zeros((dimension, num_vals), order="F")
    result += lambda1 * nodes[:, [0]]
    binom_val = 1.0
    lambda2_pow = np.ones((1, num_vals), order="F")
    for index in range(1, degree):
        lambda2_pow *= lambda2
        binom_val = (binom_val * (degree - index + 1)) / index
        result += binom_val * lambda2_pow * nodes[:, [index]]
        result *= lambda1
    result += lambda2 * lambda2_pow * nodes[:, [degree]]
    return result