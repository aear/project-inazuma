# Experience-Centered Language Learning Roadmap

## 1. Problem Diagnosis
- The current `language_processing.py` pipeline only stores hashed fragments and manually associated sound symbols, so there is no grounding between experiences and vocabulary.
- `train_from_books` slices raw text into 300-character chunks and hashes them, which feeds exposure but no semantic comprehension.
- The audio pathway records acoustic summaries without mapping them to events or transcripts.

## 2. Experience Memory Architecture
1. **Event Objects**
   - Introduce an `Event` schema stored in `memory/events/` capturing multimodal context (visual digest, audio clip, internal emotion state, action taken, resulting change).
   - Each event includes structured fields: `timestamp`, `situation_tags`, `perceived_entities`, `actions`, `outcome`, and free-form `narrative`.
2. **Episode Assembly**
   - Group temporally adjacent events into episodes and build causal links (`preconditions`, `intent`, `result`) to encode experience as sequences.
3. **Experience Graph**
   - Extend `memory_graph.py` to connect events by shared entities/emotions so Ina can traverse experiences when prompted by language.

## 3. Grounding Language in Experience
1. **Interactive Annotation Loop**
   - When a human uses a word around an event, attach it to the event entity (`"word_usage": {"speaker": "parent", "utterance": "Ina picked up the cup"}`).
   - Link words to perceptual slots (e.g., `cup` → vision digest object ID) instead of directly to sound symbols.
2. **Self-Narration**
   - After an episode concludes, run a narration routine: Ina describes the episode using available vocabulary and emotions. Store both the generated utterance and evaluator feedback for later correction.
3. **Outcome-Based Reinforcement**
   - Introduce a reward signal when narrated language causes correct human confirmation or successful action, adjusting confidence of word-event links (rather than symbol-word hashes).

## 4. Learning Mechanisms Beyond Exposure
1. **Experience Replay**
   - Schedule nightly `dreamstate` sessions to replay stored episodes, converting sensory fragments into simplified symbolic narratives to practice lexical choices.
   - Track confusion cases where vocabulary is missing and queue them as questions via `seed_self_question`.
2. **Concept Formation**
   - Cluster events by shared entities/emotions to form proto-concepts (e.g., “cup-hand-pour” cluster) and map stable clusters to candidate words supplied by humans.
3. **Embodied Simulations**
   - Use `motor_layer.py` to simulate actions described in language prompts, letting Ina predict sensory outcomes and compare with stored experiences for grounding.

## 5. Tooling Changes Needed
- Add an `experience_logger.py` module that subscribes to vision/audio/motor updates and writes event records.
- Extend `fragmentation_engine.py` to emit structured object detections instead of raw hashed fragments.
- Modify `language_processing.py` to query the experience graph when interpreting or generating utterances, prioritizing words tied to relevant episodes.
- Create evaluation scripts (`tests/test_experience_language.py`) that verify words are backed by at least one experience node before being considered “known.”

## 6. Incremental Adoption Plan
1. Prototype event schema and logging during a single interactive session.
2. Build the graph linkage and self-narration routines for those events.
3. Integrate human feedback loop to correct narrations, reinforcing accurate word grounding.
4. Expand to automated clustering and reinforcement as vocabulary grows.

Implementing these layers turns language learning into the byproduct of Ina’s lived episodes, rather than exposure to disconnected text fragments.
