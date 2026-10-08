import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# ============================================================
# PROGRAM 5
# H2 QUANTUM SIMULATION - FINAL COMPARISON
# ============================================================


# ============================================================
# 1. CREATE H2-STYLE QUANTUM CIRCUIT
# ============================================================

def create_circuit(theta):

    qc = QuantumCircuit(2, 2)

    qc.ry(theta, 0)
    qc.ry(theta, 1)

    # Entanglement
    qc.cx(0, 1)

    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


# ============================================================
# 2. CALCULATE ENERGY FROM COUNTS
# ============================================================

def calculate_energy(counts):

    shots = sum(counts.values())

    z0 = 0
    z1 = 0
    z0z1 = 0

    for state, count in counts.items():

        q1 = int(state[0])
        q0 = int(state[1])

        value_z0 = 1 if q0 == 0 else -1
        value_z1 = 1 if q1 == 0 else -1

        z0 += value_z0 * count
        z1 += value_z1 * count
        z0z1 += value_z0 * value_z1 * count

    z0 /= shots
    z1 /= shots
    z0z1 /= shots

    energy = (
        0.5 * z0
        + 0.5 * z1
        + 0.2 * z0z1
    )

    return energy


# ============================================================
# 3. EXACT ENERGY
# ============================================================

# Hamiltonian:
#
# H = 0.5 Z0 + 0.5 Z1 + 0.2 Z0Z1
#
# Matrix representation

I = np.eye(2)

Z = np.array([
    [1, 0],
    [0, -1]
])

Z0 = np.kron(Z, I)
Z1 = np.kron(I, Z)
Z0Z1 = np.kron(Z, Z)

H = (
    0.5 * Z0
    + 0.5 * Z1
    + 0.2 * Z0Z1
)

eigenvalues = np.linalg.eigvalsh(H)

exact_energy = np.min(eigenvalues)


# ============================================================
# 4. IDEAL SIMULATION
# ============================================================

ideal_simulator = AerSimulator()

theta = np.pi

circuit = create_circuit(theta)

ideal_job = ideal_simulator.run(
    circuit,
    shots=5000
)

ideal_result = ideal_job.result()

ideal_counts = ideal_result.get_counts()

ideal_energy = calculate_energy(
    ideal_counts
)


# ============================================================
# 5. NOISY SIMULATION
# ============================================================

noise_model = NoiseModel()

single_error = depolarizing_error(
    0.01,
    1
)

two_error = depolarizing_error(
    0.02,
    2
)

noise_model.add_all_qubit_quantum_error(
    single_error,
    ["ry"]
)

noise_model.add_all_qubit_quantum_error(
    two_error,
    ["cx"]
)

noisy_simulator = AerSimulator(
    noise_model=noise_model
)

noisy_job = noisy_simulator.run(
    circuit,
    shots=5000
)

noisy_result = noisy_job.result()

noisy_counts = noisy_result.get_counts()

noisy_energy = calculate_energy(
    noisy_counts
)


# ============================================================
# 6. ZERO-NOISE EXTRAPOLATION
# ============================================================

noise_levels = [1, 2, 3]

noise_energies = []

for scale in noise_levels:

    model = NoiseModel()

    error1 = depolarizing_error(
        0.01 * scale,
        1
    )

    error2 = depolarizing_error(
        0.02 * scale,
        2
    )

    model.add_all_qubit_quantum_error(
        error1,
        ["ry"]
    )

    model.add_all_qubit_quantum_error(
        error2,
        ["cx"]
    )

    simulator = AerSimulator(
        noise_model=model
    )

    job = simulator.run(
        circuit,
        shots=5000
    )

    result = job.result()

    counts = result.get_counts()

    energy = calculate_energy(counts)

    noise_energies.append(energy)


# Linear extrapolation

coefficients = np.polyfit(
    noise_levels,
    noise_energies,
    1
)

mitigated_energy = coefficients[1]


# ============================================================
# 7. ERROR CALCULATIONS
# ============================================================

ideal_error = abs(
    ideal_energy - exact_energy
)

noisy_error = abs(
    noisy_energy - exact_energy
)

mitigated_error = abs(
    mitigated_energy - exact_energy
)


# ============================================================
# 8. RESULTS TABLE
# ============================================================

print("=" * 75)
print("H2 QUANTUM SIMULATION - FINAL RESULTS")
print("=" * 75)

print(
    f"{'Method':<25}"
    f"{'Energy (Ha)':<20}"
    f"{'Error (Ha)':<20}"
)

print("-" * 75)

print(
    f"{'Exact':<25}"
    f"{exact_energy:<20.8f}"
    f"{0:<20.8f}"
)

print(
    f"{'Ideal Simulator':<25}"
    f"{ideal_energy:<20.8f}"
    f"{ideal_error:<20.8f}"
)

print(
    f"{'Noisy Simulator':<25}"
    f"{noisy_energy:<20.8f}"
    f"{noisy_error:<20.8f}"
)

print(
    f"{'ZNE Mitigated':<25}"
    f"{mitigated_energy:<20.8f}"
    f"{mitigated_error:<20.8f}"
)

print("-" * 75)


# ============================================================
# 9. ERROR IMPROVEMENT
# ============================================================

if noisy_error > 0:

    improvement = (
        (noisy_error - mitigated_error)
        / noisy_error
    ) * 100

    print(
        f"\nError improvement after mitigation: "
        f"{improvement:.2f}%"
    )


# ============================================================
# 10. GRAPH
# ============================================================

methods = [
    "Exact",
    "Ideal",
    "Noisy",
    "Mitigated"
]

energies = [
    exact_energy,
    ideal_energy,
    noisy_energy,
    mitigated_energy
]

plt.figure(figsize=(10, 6))

plt.plot(
    methods,
    energies,
    marker="o",
    linewidth=2
)

plt.ylabel("Energy (Hartree)")
plt.title("H₂ Energy Comparison")

plt.grid(True)

plt.show()


# ============================================================
# 11. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 75)
print("PROJECT CONCLUSION")
print("=" * 75)

print("""
The H2 quantum simulation was evaluated using:

1. Exact diagonalization
2. Ideal quantum simulation
3. Noisy quantum simulation
4. Zero-Noise Extrapolation

The noisy simulation demonstrates the effect of
quantum hardware errors.

Zero-Noise Extrapolation attempts to recover a
more accurate estimate of the molecular energy.
""")

print("=" * 75)