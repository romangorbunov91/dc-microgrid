import numpy as np

def compute_reward(voltages, currents, count, tau, U_ref):
    avg_U = np.mean(voltages)
    eU = abs(avg_U - U_ref) / U_ref
    I_avg = np.mean(currents)
    if abs(I_avg) < 1e-6:
        eI = 0.0
    else:
        eI = np.sum(np.abs((currents - I_avg) / I_avg))

    alpha1, beta1 = 8.0, 0.5
    alpha2, beta2 = 5.0, 1.0

    if eU <= 0.05:
        if count <= tau:
            count += 1
        if count <= tau:
            R = -alpha1 * eU - beta1 * eI
        else:
            if eI <= 0.05:
                R = alpha2 * (0.05 - eU) + beta2 * (0.5 - eI) + 0.5
            elif eI <= 0.5:
                R = alpha2 * (0.05 - eU) + beta2 * (0.5 - eI)
            else:
                R = alpha2 * (0.05 - eU) - beta1 * eI
    elif eU <= 0.2:
        R = -alpha1 * eU - beta1 * eI
        count = max(0, count - 2)
    else:
        R = -50.0
        count = 0
    return R, count