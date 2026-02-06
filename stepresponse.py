import numpy as np
from scipy.signal import StateSpace, step


def step_response(A, B, C, D, T=None):
    """Generate the step response of a linear state space system.

    The system is defined as:
        dx/dt = A x + B u
            y = C x + D u

    Parameters
    ----------
    A : array_like
        State matrix (n x n).
    B : array_like
        Input matrix (n x m).
    C : array_like
        Output matrix (p x n).
    D : array_like
        Feedthrough matrix (p x m).
    T : array_like, optional
        Time vector. If None, automatically determined.

    Returns
    -------
    t : ndarray
        Time values.
    y : ndarray
        Step response output.
    """
    sys = StateSpace(A, B, C, D)
    t, y = step(sys, T=T)
    return t, y
