from qiskit_ibm_catalog import QiskitFunctionsCatalog

tem_function_name = "algorithmiq/tem"
catalog = QiskitFunctionsCatalog(channel="ibm_cloud")

# Load your function
tem = catalog.load(tem_function_name)

print(tem)