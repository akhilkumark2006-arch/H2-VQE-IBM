# ============================================================
# PROGRAM 4 — H2 ON REAL IBM QUANTUM HARDWARE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# ============================================================
# 1. CONNECT TO IBM QUANTUM
# ============================================================

# First time only:
#
# from qiskit_ibm_runtime import QiskitRuntimeService
#
# QiskitRuntimeService.save_account(
#     channel="ibm_quantum_platform",
#     token="YOUR_IBM_QUANTUM_API_KEY",
#     overwrite=True
# )

service = QiskitRuntimeService(
    channel="ibm_quantum_platform"
)


# ============================================================
# 2. SELECT IBM QUANTUM BACKEND
# ============================================================

backend = service.least_busy(
    simulator=False,
    operational=True
)

print("=" * 60)
print("IBM QUANTUM HARDWARE")
print("=" * 60)

print("\nSelected backend:")
print(backend.name)


# ============================================================
# 3. CREATE H2-STYLE 2-QUBIT CIRCUIT
# ============================================================

theta = np.pi / 2

qc = QuantumCircuit(2)

qc.ry(theta, 0)
qc.ry(theta, 1)

qc.cx(0, 1)

qc.measure_all()

print("\nOriginal circuit:")
print(qc)


# ============================================================
# 4. TRANSPILATION FOR REAL HARDWARE
# ============================================================

pass_manager = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1
)

isa_circuit = pass_manager.run(qc)

print("\nTranspiled circuit:")
print(isa_circuit)


# ============================================================
# 5. RUN ON REAL IBM HARDWARE
# ============================================================

sampler = SamplerV2(
    mode=backend
)

job = sampler.run(
    [isa_circuit],
    shots=5000
)

print("\nJob submitted.")
print("Job ID:", job.job_id())

print("\nWaiting for IBM Quantum result...")

result = job.result()


# ============================================================
# 6. GET MEASUREMENT COUNTS
# ============================================================

pub_result = result[0]

data = pub_result.data

counts = data.meas.get_counts()


print("\nMeasurement counts:")
print(counts)


# ============================================================
# 7. CALCULATE EXPECTATION VALUES
# ============================================================

shots = sum(counts.values())

z0 = 0
z1 = 0
z0z1 = 0

for state, count in counts.items():

    # Qiskit measurement string:
    # q1 q0

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


# ============================================================
# 8. CALCULATE ENERGY
# ============================================================

energy = (
    0.5 * z0
    + 0.5 * z1
    + 0.2 * z0z1
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("REAL HARDWARE RESULTS")
print("=" * 60)

print("\nBackend:")
print(backend.name)

print("\nExpectation values:")

print("<Z0>   =", z0)
print("<Z1>   =", z1)
print("<Z0Z1> =", z0z1)

print("\nEstimated energy:")
print(
    f"{energy:.8f} Hartree"
)


# ============================================================
# 10. MEASUREMENT DISTRIBUTION
# ============================================================

states = list(counts.keys())
values = list(counts.values())

probabilities = [
    value / shots
    for value in values
]


plt.figure(figsize=(8, 5))

plt.bar(
    states,
    probabilities
)

plt.xlabel("Measured State")
plt.ylabel("Probability")

plt.title(
    "H₂ Quantum Measurement on IBM Hardware"
)

plt.grid(axis="y")

plt.show()


# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("H2 HARDWARE SIMULATION COMPLETE")
print("=" * 60)

print("Backend          :", backend.name)
print("Shots            :", shots)
print("Energy           :", energy)
print("Job ID           :", job.job_id())

print("=" * 60)