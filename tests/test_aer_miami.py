"""Smoke test: mirror ibm_miami (Nighthawk) locally with Aer and run a Bell state."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

TARGET = "ibm_miami"

service = QiskitRuntimeService()

try:
    real_backend = service.backend(TARGET)
except Exception as exc:
    available = [b.name for b in service.backends(simulator=False)]
    raise SystemExit(
        f"Could not load {TARGET!r}: {exc}\n"
        f"Backends visible to this instance: {available}"
    )

print(f"Target backend: {real_backend.name}")
print(f"  num_qubits   : {real_backend.num_qubits}")
print(f"  basis_gates  : {real_backend.operation_names}")
print(f"  processor    : {getattr(real_backend, 'processor_type', 'n/a')}")

sim = AerSimulator.from_backend(real_backend)
print(f"\nAer mirror built from {real_backend.name} calibration snapshot.")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

qc_isa = transpile(qc, backend=sim, optimization_level=3)
print(f"Transpiled depth: {qc_isa.depth()}, gates: {qc_isa.count_ops()}")

shots = 4096
result = sim.run(qc_isa, shots=shots).result()
counts = result.get_counts()

print(f"\nBell state on noisy {real_backend.name} mirror ({shots} shots):")
for bitstring in sorted(counts):
    pct = 100 * counts[bitstring] / shots
    print(f"  {bitstring}: {counts[bitstring]:>5}  ({pct:5.2f}%)")

p_correct = (counts.get("00", 0) + counts.get("11", 0)) / shots
print(f"\nFidelity proxy P(00)+P(11) = {p_correct:.4f}  (noiseless = 1.0)")
