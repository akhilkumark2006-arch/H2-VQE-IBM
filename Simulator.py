import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ============================================================
# QUANTUM SIMULATOR
# ============================================================

simulator = AerSimulator()


# ============================================================
# 1. CREATE AND RUN A 2-QUBIT PARAMETERIZED CIRCUIT
# ============================================================

def run_circuit(theta0, theta1, shots=1000):

    qc = QuantumCircuit(2, 2)

    # Parameterized rotations
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)

    # Measure both qubits
    qc.measure(0, 0)
    qc.measure(1, 1)

    print("\nQuantum circuit:")
    print(qc.draw())

    # Run on simulator
    job = simulator.run(qc, shots=shots)

    result = job.result()

    counts = result.get_counts()

    return counts


# ============================================================
# 2. CALCULATE EXPECTATION VALUES
# ============================================================

def calculate_expectations(counts):

    shots = sum(counts.values())

    z0 = 0.0
    z1 = 0.0
    z0z1 = 0.0

    for state, count in counts.items():

        # Qiskit displays two-qubit results as q1q0
        q1 = int(state[0])
        q0 = int(state[1])

        # Z eigenvalues:
        # |0> -> +1
        # |1> -> -1

        value_z0 = 1 if q0 == 0 else -1
        value_z1 = 1 if q1 == 0 else -1

        z0 += value_z0 * count
        z1 += value_z1 * count
        z0z1 += value_z0 * value_z1 * count

    # Convert to expectation values
    z0 /= shots
    z1 /= shots
    z0z1 /= shots

    return z0, z1, z0z1


# ============================================================
# 3. HAMILTONIAN ENERGY
# ============================================================

def calculate_energy(z0, z1, z0z1):

    # Educational 2-qubit Hamiltonian:
    #
    # H = 0.5 Z0 + 0.5 Z1 + 0.2 Z0Z1

    energy = (
        0.5 * z0
        + 0.5 * z1
        + 0.2 * z0z1
    )

    return energy


# ============================================================
# 4. RUN A SINGLE EXPERIMENT
# ============================================================

def run_experiment(theta0, theta1, shots=1000):

    counts = run_circuit(theta0, theta1, shots)

    z0, z1, z0z1 = calculate_expectations(counts)

    energy = calculate_energy(z0, z1, z0z1)

    return counts, z0, z1, z0z1, energy


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("2-QUBIT QUANTUM SIMULATOR")
    print("=" * 50)

    # Test parameters
    theta0 = 0
    theta1 = 0

    counts, z0, z1, z0z1, energy = run_experiment(
        theta0,
        theta1,
        shots=1000
    )

    print("\nParameters:")
    print("theta0 =", theta0)
    print("theta1 =", theta1)

    print("\nMeasurement counts:")
    print(counts)

    print("\nExpectation values:")
    print("<Z0>   =", z0)
    print("<Z1>   =", z1)
    print("<Z0Z1> =", z0z1)

    print("\nHamiltonian:")
    print("H = 0.5 Z0 + 0.5 Z1 + 0.2 Z0Z1")

    print("\nEnergy:")
    print(energy)

    print("\n" + "=" * 50)
