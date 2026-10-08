import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# ============================================================
# 1. ENERGY FUNCTION
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
# 2. CREATE CIRCUIT
# ============================================================

def create_circuit(theta):

    qc = QuantumCircuit(2, 2)

    qc.ry(theta, 0)
    qc.ry(theta, 1)

    qc.cx(0, 1)

    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


# ============================================================
# 3. CREATE NOISE MODEL
# ============================================================

def create_noise_model(noise_scale):

    noise_model = NoiseModel()

    one_qubit_error = depolarizing_error(
        0.01 * noise_scale,
        1
    )

    two_qubit_error = depolarizing_error(
        0.02 * noise_scale,
        2
    )

    noise_model.add_all_qubit_quantum_error(
        one_qubit_error,
        ["ry"]
    )

    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        ["cx"]
    )

    return noise_model


# ============================================================
# 4. RUN CIRCUIT WITH DIFFERENT NOISE LEVELS
# ============================================================

theta = np.pi / 2

noise_levels = [1, 2, 3]

energies = []

print("=" * 60)
print("ZERO-NOISE EXTRAPOLATION")
print("=" * 60)

for noise_scale in noise_levels:

    noise_model = create_noise_model(
        noise_scale
    )

    simulator = AerSimulator(
        noise_model=noise_model
    )

    circuit = create_circuit(theta)

    job = simulator.run(
        circuit,
        shots=5000
    )

    result = job.result()

    counts = result.get_counts()

    energy = calculate_energy(counts)

    energies.append(energy)

    print(
        f"Noise scale {noise_scale}: "
        f"{energy:.8f} Ha"
    )


# ============================================================
# 5. ZERO-NOISE EXTRAPOLATION
# ============================================================

x = np.array(noise_levels)
y = np.array(energies)

# Linear fit
coefficients = np.polyfit(
    x,
    y,
    1
)

slope = coefficients[0]
intercept = coefficients[1]

mitigated_energy = intercept


# ============================================================
# 6. RUN IDEAL SIMULATION
# ============================================================

ideal_simulator = AerSimulator()

ideal_circuit = create_circuit(theta)

ideal_job = ideal_simulator.run(
    ideal_circuit,
    shots=5000
)

ideal_result = ideal_job.result()

ideal_counts = ideal_result.get_counts()

ideal_energy = calculate_energy(
    ideal_counts
)


# ============================================================
# 7. ERROR CALCULATION
# ============================================================

noisy_error = abs(
    energies[0] - ideal_energy
)

mitigated_error = abs(
    mitigated_energy - ideal_energy
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ERROR MITIGATION RESULTS")
print("=" * 60)

print(
    f"\nIdeal energy      : "
    f"{ideal_energy:.8f} Ha"
)

print(
    f"Noisy energy      : "
    f"{energies[0]:.8f} Ha"
)

print(
    f"Mitigated energy  : "
    f"{mitigated_energy:.8f} Ha"
)

print(
    f"\nNoisy error       : "
    f"{noisy_error:.8f} Ha"
)

print(
    f"Mitigated error   : "
    f"{mitigated_error:.8f} Ha"
)


if noisy_error > 0:

    improvement = (
        (noisy_error - mitigated_error)
        / noisy_error
    ) * 100

    print(
        f"\nError improvement : "
        f"{improvement:.2f}%"
    )


# ============================================================
# 9. GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    s=80,
    label="Noisy Measurements"
)

fit_y = slope * x + intercept

plt.plot(
    x,
    fit_y,
    label="Extrapolation"
)

plt.axhline(
    ideal_energy,
    linestyle="--",
    label="Ideal Energy"
)

plt.scatter(
    0,
    mitigated_energy,
    s=120,
    label="Mitigated Energy"
)

plt.xlabel("Noise Scale")
plt.ylabel("Energy (Hartree)")

plt.title(
    "H₂ Zero-Noise Extrapolation"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 10. CONCLUSION
# ============================================================

print("\n" + "=" * 60)

print(
    "Zero-Noise Extrapolation estimates the energy "
    "that would be obtained at zero noise."
)

print(
    "This demonstrates how error mitigation can "
    "reduce the effect of quantum hardware noise."
)

print("=" * 60)