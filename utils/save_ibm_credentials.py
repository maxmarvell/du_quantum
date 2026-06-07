"""One-shot: save IBM Quantum credentials to ~/.qiskit/qiskit-ibm.json.

Run once with `uv run save_ibm_credentials.py`, then delete this file.
"""

from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService

token = getpass("IBM Quantum API token: ").strip()
instance = input("Instance CRN (crn:v1:...): ").strip()

QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token=token,
    instance=instance,
    name="default",
    set_as_default=True,
    overwrite=True,
)

service = QiskitRuntimeService()
backends = service.backends(operational=True)
print(f"\nSaved. {len(backends)} operational backend(s) visible:")
for b in backends:
    print(f"  - {b.name}")
