import numpy as np
import matplotlib.pyplot as plt
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
    sys = StateSpace(np.asarray(A, dtype=float), np.asarray(B, dtype=float),
                      np.asarray(C, dtype=float), np.asarray(D, dtype=float))
    t, y = step(sys, T=T)
    return t, y


if __name__ == "__main__":
    A = [[0, 1], [-1, -1]]
    B = [[0], [1]]
    C = [[1, 0]]
    D = [[0]]

    t, y = step_response(A, B, C, D, T=np.linspace(0, 12, 200))

    plt.figure()
    plt.plot(t, y)
    plt.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.title("Step Response (wn=1, zeta=0.5)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stepresponse.png", dpi=150)
    plt.show()
    print("Plot saved to stepresponse.png")
