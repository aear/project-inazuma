
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.quantum_info import Statevector
import numpy as np
import random

class QTransformer:
    def __init__(self, qubit_count=10):
        self.qubit_count = qubit_count
        self.backend = Aer.get_backend('aer_simulator')
        self.qc = QuantumCircuit(qubit_count, qubit_count)
        self.reset()

    def reset(self):
        self.qc = QuantumCircuit(self.qubit_count, self.qubit_count)
        self.qc.h(range(self.qubit_count))  # Put all qubits in superposition

    def inject_symbol_emotion(self, symbol_hash, emotion_vector):
        # Emotion vector is assumed to be 24 values in [-1, 1]
        seed = sum(ord(c) for c in symbol_hash) % 10000
        random.seed(seed)
        for i in range(min(self.qubit_count, len(emotion_vector))):
            angle = (emotion_vector[i] + 1) * np.pi  # map [-1, 1] to [0, 2π]
            self.qc.ry(angle, i)
            if random.random() > 0.5:
                self.qc.rz(random.random() * 2 * np.pi, i)

    def entangle_logic(self):
        for i in range(self.qubit_count - 1):
            self.qc.cx(i, i + 1)
        # Add a final mixing layer
        self.qc.h(range(self.qubit_count))

    def run_dreamstep(self):
        self.entangle_logic()
        self.qc.measure(range(self.qubit_count), range(self.qubit_count))
        transpiled = transpile(self.qc, self.backend)
        qobj = assemble(transpiled, shots=1)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        collapsed_state = list(counts.keys())[0]
        return collapsed_state

    def collapse_to_meaning(self, collapsed_state):
        # Interpret the collapsed state as three symbolic outcomes
        tag_bits = collapsed_state[:3]
        question_bits = collapsed_state[3:6]
        word_bits = collapsed_state[6:]

        tag_map = {
            '000': ['calm', 'clarity'],
            '001': ['grief', 'echo'],
            '010': ['hope', 'unknown'],
            '011': ['tension', 'shift'],
            '100': ['trust', 'fire'],
            '101': ['betrayal', 'ice'],
            '110': ['curiosity', 'glow'],
            '111': ['loss', 'awakening']
        }

        questions = {
            '000': "What am I avoiding?",
            '001': "Why did this feel heavy?",
            '010': "What pattern is surfacing?",
            '011': "Do I need closure?",
            '100': "Is this intuition or fear?",
            '101': "Was I wrong about them?",
            '110': "What else could this mean?",
            '111': "Is something waking up in me?"
        }

        words = {
            '000': "refuge",
            '001': "wound",
            '010': "spark",
            '011': "ghost",
            '100': "veil",
            '101': "pulse",
            '110': "womb",
            '111': "echo"
        }

        tag_cluster = tag_map.get(tag_bits, ["unknown"])
        question = questions.get(question_bits, "What was I feeling?")
        word = words.get(word_bits, "???")

        return {
            "tags": tag_cluster,
            "self_question": question,
            "poetic_word": word,
            "raw_bits": collapsed_state
        }

    def dream(self, symbol, emotion_vector):
        self.reset()
        self.inject_symbol_emotion(symbol, emotion_vector)
        collapsed = self.run_dreamstep()
        return self.collapse_to_meaning(collapsed)
