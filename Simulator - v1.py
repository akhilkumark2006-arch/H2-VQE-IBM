import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA


# ============================================================
# 1. CREATE H2 MOLECULE
# ============================================================

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0
)

problem = driver.run()

print("=" * 60)
print("H2 MOLECULE SIMULATION")
print("=" * 60)

print("\nNumber of spatial orbitals:", problem.num_spatial_orbitals)
print("Number of particles:", problem.num_particles)


# ============================================================
# 2. CONVERT MOLECULAR HAMILTONIAN TO QUBIT HAMILTONIAN
# ============================================================

mapper = ParityMapper()

hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

print("\nQubit Hamiltonian:")
print(hamiltonian)


# ============================================================
# 3. CREATE PARAMETERIZED QUANTUM CIRCUIT
# ============================================================

ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    qubit_mapper=mapper
)

print("\nQuantum circuit:")
print(ansatz.draw())


# ============================================================
# 4. QUANTUM SIMULATOR
# ============================================================

backend = AerSimulator()


# ============================================================
# 5. VQE OPTIMIZATION
# ============================================================

energy_history = []


def callback(eval_count, parameters, mean, metadata):
    energy_history.append(mean)

    print(
        f"Iteration {eval_count:3d} | "
        f"Energy = {mean:.10f} Ha"
    )


optimizer = SPSA(
    maxiter=100
)


vqe = VQE(
    estimator=backend,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback
)


# ============================================================
# 6. RUN VQE
# ============================================================

print("\nRunning VQE...\n")

solver = GroundStateEigensolver(
    mapper,
    vqe
)

result = solver.solve(problem)


# ============================================================
# 7. VQE RESULT
# ============================================================

vqe_energy = result.total_energies[0].real

print("\n" + "=" * 60)
print("VQE RESULT")
print("=" * 60)

print(f"\nGround-state energy: {vqe_energy:.10f} Hartree")


# ============================================================
# 8. EXACT REFERENCE ENERGY
# ============================================================

exact_solver = GroundStateEigensolver(
    mapper,
    VQE(
        estimator=backend,
        ansatz=ansatz,
        optimizer=SPSA(maxiter=1)
    )
)

# Exact diagonalization of the qubit Hamiltonian
matrix = hamiltonian.to_matrix()

eigenvalues = np.linalg.eigvalsh(matrix)

exact_energy = np.min(eigenvalues).real


print("\nExact reference energy:")
print(f"{exact_energy:.10f} Hartree")


# ============================================================
# 9. ERROR
# ============================================================

error = abs(vqe_energy - exact_energy)

print("\nAbsolute error:")
print(f"{error:.10f} Hartree")


if exact_energy != 0:
    percentage_error = (
        abs(vqe_energy - exact_energy)
        / abs(exact_energy)
    ) * 100

    print(f"Percentage error: {percentage_error:.4f}%")


# ============================================================
# 10. CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(energy_history) + 1),
    energy_history,
    marker="o",
    markersize=3
)

plt.axhline(
    exact_energy,
    linestyle="--",
    label="Exact Energy"
)

plt.xlabel("VQE Iteration")
plt.ylabel("Energy (Hartree)")
plt.title("H₂ VQE Energy Convergence")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Molecule       : H₂")
print(f"Bond distance  : 0.735 Å")
print(f"Basis set      : STO-3G")
print(f"VQE energy     : {vqe_energy:.10f} Ha")
print(f"Exact energy   : {exact_energy:.10f} Ha")
print(f"Absolute error : {error:.10f} Ha")
print("=" * 60)