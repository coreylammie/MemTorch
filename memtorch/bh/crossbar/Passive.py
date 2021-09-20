import numpy as np
import torch

import memtorch

if "cpu" not in memtorch.__version__:
    import memtorch_cuda_bindings

import memtorch_bindings


def solve_passive(
    conductance_matrix,
    V_WL,
    V_BL,
    R_source,
    R_line,
    n_input_batches=None,
    det_readout_currents=True,
    use_bindings=True,
    cuda_malloc_heap_size=None,
):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    assert R_source != 0 or R_line != 0, "R_source or R_line must be non-zero."
    assert R_source >= 0 and R_line >= 0, "R_source and R_line must be >=0."
    if use_bindings:
        if n_input_batches is None:
            if "cpu" in memtorch.__version__:
                return memtorch_bindings.solve_passive(
                    conductance_matrix,
                    V_WL,
                    V_BL,
                    R_source,
                    R_line,
                    det_readout_currents=det_readout_currents,
                )
            else:
                if cuda_malloc_heap_size is None:
                    return memtorch_cuda_bindings.solve_passive(
                        conductance_matrix,
                        V_WL,
                        V_BL,
                        R_source,
                        R_line,
                        det_readout_currents=det_readout_currents,
                    )
                else:
                    return memtorch_cuda_bindings.solve_passive(
                        conductance_matrix,
                        V_WL,
                        V_BL,
                        R_source,
                        R_line,
                        det_readout_currents=det_readout_currents,
                        cuda_malloc_heap_size=cuda_malloc_heap_size,
                    )
        else:
            return memtorch_bindings.solve_passive(
                conductance_matrix.cpu(),
                V_WL.cpu(),
                V_BL.cpu(),
                R_source,
                R_line,
                n_input_batches=n_input_batches,
            ).to(device)
    else:
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
            if R_source == 0:
                values[index] = conductance_matrix[i, 0] + 1 / R_line
            elif R_line == 0:
                values[index] = conductance_matrix[i, 0] + 1 / R_source
            else:
                values[index] = conductance_matrix[i, 0] + 1 / R_source + 1 / R_line

            index += 1
            indices[0, index] = i * n + 1
            indices[1, index] = i * n
            if R_line == 0:
                values[index : index + 2] = 0
            else:
                values[index : index + 2] = -1 / R_line

            index += 1
            indices[0, index] = i * n
            indices[1, index] = i * n + 1
            index += 1
            indices[0:2, index] = i * n + (n - 1)
            if R_line == 0:
                values[index] = conductance_matrix[i, n - 1]
            else:
                values[index] = conductance_matrix[i, n - 1] + 1 / R_line

            index += 1
            for j in range(1, n - 1):
                indices[0:2, index] = i * n + j
                if R_line == 0:
                    values[index] = conductance_matrix[i, j]
                else:
                    values[index] = conductance_matrix[i, j] + 2 / R_line
                index += 1
                indices[0, index] = i * n + j + 1
                indices[1, index] = i * n + j
                if R_line == 0:
                    values[index : index + 2] = 0
                else:
                    values[index : index + 2] = -1 / R_line

                index += 1
                indices[0, index] = i * n + j
                indices[1, index] = i * n + j + 1
                index += 1

        # B matrix
        indices[0, index : index + (m * n)] = mn_range
        indices[1, index : index + (m * n)] = (
            indices[0, index : index + (m * n)] + m * n
        )
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
            if R_line == 0:
                values[index] = -conductance_matrix[0, j]
            else:
                values[index] = -1 / R_line - conductance_matrix[0, j]

            index += 1
            indices[0, index] = m * n + (j * m)
            indices[1, index] = m * n + j + n
            if R_line == 0:
                values[index : index + 2] = 0
            else:
                values[index : index + 2] = 1 / R_line

            index += 1
            indices[0, index : index + 2] = m * n + (j * m) + m - 1
            indices[1, index] = m * n + (n * (m - 2)) + j
            index += 1
            indices[1, index] = m * n + (n * (m - 1)) + j
            if R_source == 0:
                values[index] = -conductance_matrix[m - 1, j] - 1 / R_line
            elif R_line == 0:
                values[index] = -1 / R_source - conductance_matrix[m - 1, j]
            else:
                values[index] = (
                    -1 / R_source - conductance_matrix[m - 1, j] - 1 / R_line
                )

            index += 1
            indices[0, index : index + 3 * (m - 2)] = (
                m * n + (j * m) + m_range[1:-1].repeat_interleave(3)
            )
            for i in range(1, m - 1):
                indices[1, index] = m * n + (n * (i - 1)) + j
                if R_line == 0:
                    values[index : index + 2] = 0
                else:
                    values[index : index + 2] = 1 / R_line

                index += 1
                indices[1, index] = m * n + (n * (i + 1)) + j
                index += 1
                indices[1, index] = m * n + (n * i) + j
                if R_line == 0:
                    values[index] = -conductance_matrix[i, j]
                else:
                    values[index] = -conductance_matrix[i, j] - 2 / R_line

                index += 1

        if n_input_batches is None:
            E_matrix = torch.zeros(2 * m * n, device=device)
            if R_source == 0:
                E_matrix[m_range * n] = V_WL.to(device)  # E_W values
                E_matrix[m * n + (n_range + 1) * m - 1] = -V_BL.to(device)  # E_B values
            else:
                # E_W values
                E_matrix[m_range * n] = V_WL.to(device) / R_source
                E_matrix[m * n + (n_range + 1) * m - 1] = (
                    -V_BL.to(device) / R_source
                )  # E_B values

            V = torch.linalg.solve(
                torch.sparse_coo_tensor(
                    indices, values, (2 * m * n, 2 * m * n), device=device
                ).to_dense(),
                E_matrix,
            )
            V_applied = torch.zeros((m, n), device=device)
            for i in m_range:
                V_applied[i, n_range] = V[n * i + n_range] - V[m * n + n * i + n_range]
            if not det_readout_currents:
                return V_applied
            else:
                return torch.sum(torch.mul(V_applied, conductance_matrix.to(device)), 0)
        else:
            out = torch.zeros(n_input_batches, n, device=device)
            for i in range(n_input_batches):
                E_matrix = torch.zeros(2 * m * n, device=device)
                if R_source == 0:
                    E_matrix[m_range * n] = V_WL[i, :].to(device)  # E_W values
                    E_matrix[m * n + (n_range + 1) * m - 1] = -V_BL[i, :].to(
                        device
                    )  # E_B values
                else:
                    E_matrix[m_range * n] = (
                        V_WL[i, :].to(device) / R_source
                    )  # E_W values
                    E_matrix[m * n + (n_range + 1) * m - 1] = (
                        -V_BL[i, :].to(device) / R_source
                    )  # E_B values

                V = torch.linalg.solve(
                    torch.sparse_coo_tensor(
                        indices, values, (2 * m * n, 2 * m * n), device=device
                    ).to_dense(),
                    E_matrix,
                )
                V_applied = torch.zeros((m, n), device=device)
                for j in m_range:
                    V_applied[j, n_range] = (
                        V[n * j + n_range] - V[m * n + n * j + n_range]
                    )

                out[i, :] = torch.sum(
                    torch.mul(V_applied, conductance_matrix.to(device)), 0
                )

            return out
