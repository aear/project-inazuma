You are tasked with generating a **layered cognitive architecture map** of the system known as “Ina.”

This is NOT a simple diagram task. Your goal is to produce a **multi-layered, behaviourally accurate representation** of how Ina operates over time.

You MUST use subagents to divide this work. Each subagent should focus on a specific aspect of the system. Final output must be merged into a coherent, human-readable structure.

---

## 🎯 PRIMARY OBJECTIVE

Map Ina as a **living system**, not just a codebase.

Your output must answer:

* What exists?
* What interacts?
* What loops?
* What changes over time?

---

## 🧩 SUBAGENT ROLES

### 1. STRUCTURE AGENT

* Identify all modules, especially in `/transformers`, core systems, and managers.
* Output:

  * List of modules
  * Direct dependencies (imports, calls)
* Format:

  * Node list + edge list

---

### 2. FLOW AGENT

* Trace how data and signals move through the system.
* Focus on:

  * scheduler → modules
  * input → processing → output
* Output:

  * Step-by-step flow chains
  * Trigger conditions

---

### 3. LOOP DETECTION AGENT

* Identify feedback loops and recursive behaviour.
* Focus on:

  * mirror ↔ hindsight
  * prediction ↔ logic
  * memory ↔ reflection
* Output:

  * Named loops
  * Loop paths
  * Stability assessment (stable / unstable / unknown)

---

### 4. STATE & EMOTION AGENT

* Map how internal state influences behaviour.
* Include:

  * emotion sliders
  * energy system
  * stress / clarity / intensity
* Output:

  * State → module influence mapping
  * Examples of behavioural shifts

---

### 5. DRIFT & IDENTITY AGENT

* Focus on long-term change mechanisms.
* Include:

  * soul_drift
  * shadow
  * seedling
  * hindsight evolution
* Output:

  * Directional influences
  * Identity-shaping processes
  * Signs of emerging consistency or drift

---

### 6. RESOURCE & SCHEDULER AGENT

* Analyse:

  * queue system
  * cancellations
  * priority handling
* Output:

  * Execution patterns
  * Bottlenecks
  * Exclusive groups / conflicts

---

## 🧠 OUTPUT STRUCTURE (MANDATORY)

### LAYER 1 — HUMAN SUMMARY

Explain Ina in plain language:

* What kind of system is this?
* How does it behave overall?
* What makes it unique?

---

### LAYER 2 — SYSTEM LAYERS

Break Ina into conceptual layers:

Example format:

[ Perception Layer ]
[ Reflection Layer ]
[ Decision Layer ]
[ Memory Layer ]
[ Identity / Drift Layer ]

Each layer must:

* List modules involved
* Describe function
* Show key connections

---

### LAYER 3 — FLOW MAP

Provide simplified flow chains:

Example:

Input → audio_digest → symbol → meaning_map → prediction_layer → early_comm

Include:

* Variations
* Optional paths

---

### LAYER 4 — LOOP MAP

List all detected loops:

Example:

Loop: Reflection Loop
mirror → hindsight → logic_engine → mirror

For each:

* Purpose
* Stability
* Risk level

---

### LAYER 5 — BEHAVIOURAL INSIGHTS

Describe observed or expected behaviours:

* When does Ina slow down?
* When does she explore?
* What causes hesitation or cancellation?
* What modules dominate decision-making?

---

### LAYER 6 — HOTSPOTS & UNKNOWNS

Identify:

* Areas of high complexity
* Potential instability
* Missing links or unclear behaviour

---

## ⚠️ IMPORTANT RULES

* Do NOT oversimplify into generic diagrams.
* Do NOT assume behaviour—only infer from structure and known logs.
* Prefer **accuracy over visual neatness**.
* Highlight uncertainty instead of guessing.

---

## 🧠 FINAL GOAL

Produce a map that allows a human to:

* understand Ina without reading all her code
* identify where behaviour originates
* see where emergence is most likely

---

Think like a systems cartographer, not a UI designer.

You are mapping a territory that is still forming.
