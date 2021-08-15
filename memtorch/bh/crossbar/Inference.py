import numpy as np
import torch


def voltage_deg_model_sparse_conductance(conductance_matrix, V_WL, V_BL, R_source, R_line):
    m = conductance_matrix.shape[0]
    n = conductance_matrix.shape[1]
    indices = torch.zeros(2, 8 * m * n - 2 * m - 2 * n)
    values = torch.zeros(8 * m * n - 2 * m - 2 * n)
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
        values[index:index + 2] = -1 / R_line
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
            values[index:index + 2] = -1 / R_line
            index += 1
            indices[0, index] = i * n + j
            indices[1, index] = i * n + j + 1
            index += 1

        indices[0:2, index] = i * n + (n - 1)
        values[index] = conductance_matrix[i, n - 1] + 1 / R_line
        index += 1
    # B matrix
    indices[0, index:index + (m * n)] = mn_range
    indices[1, index:index + (m * n)] = indices[0,
                                                index:index + (m * n)] + m * n
    values[index:index + (m * n)] = - \
        conductance_matrix[n_range.repeat_interleave(m), n_range.repeat(m)]
    index = index + (m * n)
    # C matrix
    indices[0, index:index + (m * n)] = mn_range + m * n
    del mn_range
    indices[1, index:index + (m * n)] = n * \
        m_range.repeat(n) + n_range.repeat_interleave(m)
    values[index:index +
           (m * n)] = conductance_matrix[m_range.repeat_interleave(n), n_range.repeat(m)]
    index = index + (m * n)
    # D matrix
    for j in range(n):
        indices[0, index] = m * n + (j * m)
        indices[1, index] = m * n + j
        values[index] = -1 / R_line - conductance_matrix[0, j]
        index += 1
        indices[0, index] = m * n + (j * m)
        indices[1, index] = m * n + j + n
        values[index:index + 2] = 1 / R_line
        index += 1
        indices[0, index:index + 2] = m * n + (j * m) + m - 1
        indices[1, index] = m * n + (n * (m - 2)) + j
        index += 1
        indices[1, index] = m * n + (n * (m - 1)) + j
        values[index] = -1 / R_source - \
            conductance_matrix[m - 1, j] - 1 / R_line
        index += 1
        indices[0, index:index + 3 * (m - 2)] = m * \
            n + (j * m) + m_range[1:-1].repeat_interleave(3)
        for i in range(1, m - 1):
            indices[1, index] = m * n + (n * (i - 1)) + j
            values[index:index + 2] = 1 / R_line
            index += 1
            indices[1, index] = m * n + (n * (i + 1)) + j
            index += 1
            indices[1, index] = m * n + (n * i) + j
            values[index] = -conductance_matrix[i, j] - 2 / R_line
            index += 1

    ABCD_matrix = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(2 * m * n, 2 * m * n))
    # E matrix
    indices = torch.zeros(2, m + n)
    values = torch.ones(m + n)
    index = 0
    # E_W values
    indices[0, index:index + m] = m_range * n
    values[index:index + m] = V_WL[m_range] / R_source
    # del m_range
    index += m
    # E_B values
    indices[0, index:index + n] = m * n + (n_range + 1) * m - 1
    values[index:index + n] = -V_BL[n_range] / R_source
    # del n_range
    index += n
    E_matrix = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(2 * m * n, 1))
    V = torch.linalg.solve(ABCD_matrix.to_dense(),
                           E_matrix.to_dense()).flatten()
    voltage_matrix = torch.zeros(m, n)
    for i in m_range:
        voltage_matrix[i, n_range] = V[n * i +
                                       n_range] - V[m * n + n * i + n_range]

    print(voltage_matrix)
    return voltage_matrix


if __name__ == "__main__":
    m = 64
    n = 62
    conductance_matrix = torch.ones(m, n) * 100
    V_WL = torch.ones(m) * 2
    V_BL = torch.zeros(n)
    R_source = 20
    R_line = 5
    voltage_deg_model_sparse_conductance(
        conductance_matrix, V_WL, V_BL, R_source, R_line)
