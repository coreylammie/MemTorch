import numpy as np
import torch

import memtorch
import memtorch_bindings


def naive_inference_passive(
    conductance_matrix, V_WL, V_BL, R_source, R_line, return_current=True
):
    # device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    device = torch.device("cpu")
    m = conductance_matrix.shape[0]
    n = conductance_matrix.shape[1]
    indices = torch.zeros(2, 8 * m * n - 2 * m - 2 * n, device=device)
    values = torch.zeros(8 * m * n - 2 * m - 2 * n, device=device)
    mn_range = torch.arange(m * n)
    m_range = torch.arange(m)
    n_range = torch.arange(n)
    index = 0
    # A matrix
    for i in range(m):
        indices[0:2, index] = i * n
        values[index] = conductance_matrix[i, 0] + 1 / R_source + 1 / R_line
        index += 1
        indices[0, index] = i * n + 1
        indices[1, index] = i * n
        values[index : index + 2] = -1 / R_line
        index += 1
        indices[0, index] = i * n
        indices[1, index] = i * n + 1
        index += 1
        for j in range(1, n - 1):
            indices[0:2, index] = i * n + j
            values[index] = conductance_matrix[i, j] + 2 / R_line
            index += 1
            indices[0, index] = i * n + j + 1
            indices[1, index] = i * n + j
            values[index : index + 2] = -1 / R_line
            index += 1
            indices[0, index] = i * n + j
            indices[1, index] = i * n + j + 1
            index += 1

        indices[0:2, index] = i * n + (n - 1)
        values[index] = conductance_matrix[i, n - 1] + 1 / R_line
        index += 1
    # B matrix
    indices[0, index : index + (m * n)] = mn_range
    indices[1, index : index + (m * n)] = indices[0, index : index + (m * n)] + m * n
    values[index : index + (m * n)] = -conductance_matrix[
        n_range.repeat_interleave(m), n_range.repeat(m)
    ]
    index = index + (m * n)
    # C matrix
    indices[0, index : index + (m * n)] = mn_range + m * n
    del mn_range
    indices[1, index : index + (m * n)] = n * m_range.repeat(
        n
    ) + n_range.repeat_interleave(m)
    values[index : index + (m * n)] = conductance_matrix[
        m_range.repeat_interleave(n), n_range.repeat(m)
    ]
    index = index + (m * n)
    # D matrix
    for j in range(n):
        indices[0, index] = m * n + (j * m)
        indices[1, index] = m * n + j
        values[index] = -1 / R_line - conductance_matrix[0, j]
        index += 1
        indices[0, index] = m * n + (j * m)
        indices[1, index] = m * n + j + n
        values[index : index + 2] = 1 / R_line
        index += 1
        indices[0, index : index + 2] = m * n + (j * m) + m - 1
        indices[1, index] = m * n + (n * (m - 2)) + j
        index += 1
        indices[1, index] = m * n + (n * (m - 1)) + j
        values[index] = -1 / R_source - conductance_matrix[m - 1, j] - 1 / R_line
        index += 1
        indices[0, index : index + 3 * (m - 2)] = (
            m * n + (j * m) + m_range[1:-1].repeat_interleave(3)
        )
        for i in range(1, m - 1):
            indices[1, index] = m * n + (n * (i - 1)) + j
            values[index : index + 2] = 1 / R_line
            index += 1
            indices[1, index] = m * n + (n * (i + 1)) + j
            index += 1
            indices[1, index] = m * n + (n * i) + j
            values[index] = -conductance_matrix[i, j] - 2 / R_line
            index += 1

    E_matrix = torch.zeros(2 * m * n)
    E_matrix[m_range * n] = V_WL[m_range] / R_source  # E_W values
    E_matrix[m * n + (n_range + 1) * m - 1] = -V_BL[n_range] / R_source  # E_B values
    V = memtorch_bindings.solve_sparse_linear(
        indices[0], indices[1], values, (2 * m * n, 2 * m * n), E_matrix
    )
    voltage_matrix = torch.zeros((m, n), device=device)
    for i in m_range:
        voltage_matrix[i, n_range] = V[n * i + n_range] - V[m * n + n * i + n_range]

    if return_current:
        return torch.sum(torch.mul(voltage_matrix, conductance_matrix), 0)
    else:
        return voltage_matrix


if __name__ == "__main__":
    m = 64
    n = 62
    conductance_matrix = torch.ones(m, n) * 100
    V_WL = torch.ones(m) * 2
    V_BL = torch.zeros(n)
    R_source = 20
    R_line = 5
    out = naive_inference_passive(conductance_matrix, V_WL, V_BL, R_source, R_line)
    print(out)
    print(out.shape)
