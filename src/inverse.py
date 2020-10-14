import numpy as np


def invert_matrix(A: np.ndarray) -> np.ndarray:
    "Inverts a matrix using Gaussian - Jordan elimination"
    m, n = A.shape

    # append identity matrix (now m x 2n)
    A = np.concatenate([A, np.eye(m, n)], axis=-1)

    rref = reduced_row_echelon_form(A, m, n)
    return rref[:, n:]


def reduced_row_echelon_form(A, num_rows, num_columns):
    """"Reduced row echelon form algorithm taken from
    https://en.wikipedia.org/wiki/Row_echelon_form
    """
    # lead column which we cycle through
    lead = 0
    for r in range(num_rows):
        if num_columns <= lead:
            break
        i = r
        while A[i, lead] == 0:
            i += 1
            if i == num_rows:
                # we need move lead ahead
                i = r
                lead += 1
                if num_columns == lead:
                    break

        A[[i, r]] = A[[r, i]]
        if A[r, lead] != 0:
            A[r] /= A[r, lead]

        for i in range(num_rows):
            if i != r:
                A[i] -= A[r] * A[i, lead]

        lead += 1

    return A
