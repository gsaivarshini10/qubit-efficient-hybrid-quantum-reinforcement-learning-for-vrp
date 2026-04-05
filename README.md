# qubit-efficient-hybrid-quantum-reinforcement-learning-for-vrp


📌 Overview

This project explores the application of Quantum Reinforcement Learning (QRL) to solve the Vehicle Routing Problem (VRP) — a well-known NP-hard combinatorial optimization problem.

Unlike traditional approaches that rely on exhaustive or heuristic search, our method leverages parameterized quantum circuits to learn optimal routing policies, reducing the effective decision complexity.

🎯 Objectives
Implement a Quantum Reinforcement Learning model using PennyLane
Encode routing states into quantum circuits
Train a policy using gradient-based optimization
Compare performance with Grover’s search algorithm
Demonstrate improved scaling behavior
🧠 Key Idea

Instead of searching through all possible routes, the model:

Learns an optimal routing policy
Uses quantum superposition and entanglement
Reduces repeated search operations
⚙️ Quantum Circuit Design
Qubits: 6
Encoding: Rotation gates (RY)
Trainable Layers: RY, RZ, RX
Entanglement: CNOT ring structure
Measurement: Expectation value of Pauli-Z
📊 Time Complexity Analysis
Method	Complexity
Classical Search	O(N)
Grover’s Algorithm	O(√N)
Proposed QRL	O(n) ≈ O(log N)

Note: QRL does not reduce the theoretical complexity of VRP (NP-hard), but significantly reduces effective decision complexity through learning.

📈 Results
1. Complexity Comparison
Grover: √N growth
QRL: Logarithmic-like growth
2. Training Convergence
Loss decreases over iterations
Demonstrates learning capability
🛠️ Tech Stack
Python
PennyLane (Quantum ML)
PyTorch (for optimization)
NumPy
Matplotlib
