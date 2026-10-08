import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ============================================================
# 1. CREATE NOISE MODEL
# ============================================================

noise_model = NoiseModel()

# Single-qubit gate error
single_qubit_error = depolarizing_error(
    0.01,
    1
)

# Two-qubit gate error
two_qubit_error = depolarizing_error(
    0.02,
    2
)

noise_model.add_all_qubit_quantum_error(
    single_qubit_error,
    ["ry"]
)

noise_model.add_all_qubit_quantum_error(
    two_qubit_error,
    ["cx"]
)


# ============================================================
# 2. CREATE SIMULATORS
# ============================================================

ideal_simulator = AerSimulator()

noisy_simulator = AerSimulator(
    noise_model=noise_model
)


# ============================================================
# 3. CREATE H2-LIKE 2-QUBIT CIRCUIT
# ============================================================

def create_circuit(theta0, theta1):

    qc = QuantumCircuit(2, 2)

    # Parameterized rotations
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)

    # Entanglement
    qc.cx(0, 1)

    # Measurement
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


# ============================================================
# 4. CALCULATE EXPECTATION VALUES
# ============================================================

def calculate_expectations(counts):

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

    return z0, z1, z0z1


# ============================================================
# 5. CALCULATE ENERGY
# ============================================================

def calculate_energy(z0, z1, z0z1):

    return (
        0.5 * z0
        + 0.5 * z1
        + 0.2 * z0z1
    )


# ============================================================
# 6. RUN SIMULATION
# ============================================================

theta_values = np.linspace(
    0,
    np.pi,
    20
)

ideal_energies = []
noisy_energies = []


for theta in theta_values:

    circuit = create_circuit(
        theta,
        theta
    )

    # --------------------------------------------------------
    # Ideal simulation
    # --------------------------------------------------------

    ideal_job = ideal_simulator.run(
        circuit,
        shots=2000
    )

    ideal_result = ideal_job.result()

    ideal_counts = ideal_result.get_counts()

    z0, z1, z0z1 = calculate_expectations(
        ideal_counts
    )

    ideal_energy = calculate_energy(
        z0,
        z1,
        z0z1
    )

    ideal_energies.append(
        ideal_energy
    )


    # --------------------------------------------------------
    # Noisy simulation
    # --------------------------------------------------------

    noisy_job = noisy_simulator.run(
        circuit,
        shots=2000
    )

    noisy_result = noisy_job.result()

    noisy_counts = noisy_result.get_counts()

    z0, z1, z0z1 = calculate_expectations(
        noisy_counts
    )

    noisy_energy = calculate_energy(
        z0,
        z1,
        z0z1
    )

    noisy_energies.append(
        noisy_energy
    )


# ============================================================
# 7. FIND BEST ENERGY
# ============================================================

ideal_min = min(ideal_energies)
noisy_min = min(noisy_energies)

ideal_index = np.argmin(ideal_energies)
noisy_index = np.argmin(noisy_energies)

ideal_theta = theta_values[ideal_index]
noisy_theta = theta_values[noisy_index]


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("H2 QUANTUM SIMULATION WITH NOISE")
print("=" * 60)

print("\nIdeal Simulation")
print("----------------")
print("Best theta:", ideal_theta)
print("Minimum energy:", ideal_min)

print("\nNoisy Simulation")
print("----------------")
print("Best theta:", noisy_theta)
print("Minimum energy:", noisy_min)

print("\nNoise Error")
print("-----------")

error = abs(
    noisy_min - ideal_min
)

print("Absolute energy error:", error)

percentage_error = (
    error /
    abs(ideal_min)
) * 100

print(
    "Percentage error:",
    percentage_error,
    "%"
)


# ============================================================
# 9. PLOT COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    theta_values,
    ideal_energies,
    marker="o",
    label="Ideal Simulation"
)

plt.plot(
    theta_values,
    noisy_energies,
    marker="x",
    label="Noisy Simulation"
)

plt.xlabel("Theta")
plt.ylabel("Energy")
plt.title("H₂ Energy: Ideal vs Noisy Quantum Simulation")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print(
    "Quantum noise changes the measured expectation values "
    "and therefore affects the estimated molecular energy."
)

print(
    "This demonstrates the difference between ideal "
    "quantum simulation and realistic noisy hardware."
)

print("=" * 60)