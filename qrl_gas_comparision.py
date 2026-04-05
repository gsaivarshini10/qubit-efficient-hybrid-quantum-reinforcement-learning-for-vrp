import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

# Quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# QRL Circuit (Parameterized)
@qml.qnode(dev)
def qrl_circuit(params):
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    
    # simple entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Reward function (maximize expectation values)
def reward(params):
    output = qrl_circuit(params)
    return -np.sum(output)  # minimize negative = maximize output

# Training loop (QRL learning)
def train_qrl(steps=50):
    params = np.random.randn(n_qubits, requires_grad=True)
    lr = 0.1
    loss_history = []

    for step in range(steps):
        grad = qml.grad(reward)(params)
        params = params - lr * grad
        
        loss = reward(params)
        loss_history.append(loss)

    return params, loss_history

# Run training
trained_params, loss_history = train_qrl()

# -------- Complexity Comparison --------
N_values = [100, 500, 1000, 5000, 10000]

grover_steps = [np.sqrt(N) for N in N_values]
qrl_steps = [np.log2(N) for N in N_values]

# -------- Graph 1: Complexity --------
plt.figure()
plt.plot(N_values, grover_steps, marker='o', label="Grover O(√N)")
plt.plot(N_values, qrl_steps, marker='o', label="QRL O(log N)")
plt.xlabel("Input Size (N)")
plt.ylabel("Steps")
plt.title("Grover vs QRL Complexity")
plt.legend()
plt.grid()
plt.show()

# -------- Graph 2: Training --------
plt.figure()
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("QRL Training Convergence")
plt.grid()
plt.show()
