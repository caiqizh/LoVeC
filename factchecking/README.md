## Overview

### Key Scripts

1. **`factchecker_sentence.py`**
   This script performs fact-checking at the sentence level within a paragraph (not at the atomic fact level).

2. **`generate_atomic_fact.py`**  
   This script generates atomic facts from a given paragraph. It supports multiple models, including `gpt-4o`, `gpt-4o mini`, and `gemini 2 flash`.  
   **Note:** Different models may produce slightly different results when breaking down atomic facts. For consistency, it is recommended to use a single model.

3. **`factchecker.py`**  
   This script includes information retrieval modules (`wiki_retrieval.py` and `wild_retrieval.py`).  
   **Note:** These modules are currently not required.


### Important Reminder
- Before running any API, ensure you estimate the cost to avoid unexpected charges.
    - You can modify my ```estimate_overall_cost`` function.

- For the current stage, only **`factchecker_sentence.py`** is needed.