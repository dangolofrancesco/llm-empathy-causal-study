# llm-empathy-causal-study
With this project we aim to determine if a more empathetic chatbot response causally increases a user’s immediate emotional attachment.

# Research Project Causal Model

This diagram outlines the causal relationships, confounders, moderators, and mediators in our study.

## Causal Model (DAG)
### 1. The Core Causal Path: Empathy $\to$ Attachment

Your fundamental question is about the direct effect of the **Treatment (T)**, which is the **LLM Empathy Score** (measured using your LLM-as-a-Judge), on the **Outcome (Y)**, which is the **User Attachment Score** (also measured using your LLM-as-a-Judge).
* **T (LLM Empathy Score)** $\to$ **Y (User Attachment Score)**

---
### 2. Confounding Variables (X): The Back-Doors to Close 

These variables create spurious correlations by affecting both T and Y. The primary challenge, as highlighted by the `ACL 2020` review, is that **text itself** is a major confounder. Our matching strategy is designed to close these back-door paths.

* **X1: User's Initial Prompt (Text Confounder)**: This is the most critical confounder. The content and emotion of the user's prompt directly cause both the LLM's likely empathy level (T) and the user's subsequent attachment (Y).
    * **Control Method:** **Text Matching** based on embeddings, as detailed in `AJPS 2020`. We are essentially finding pairs with the same X1 but different T to isolate the T $\to$ Y effect. The `ICML 2016` paper provides the theoretical basis for using learned representations (embeddings) for this counterfactual task.
* **X2: User's Latent Traits (Unseen Confounder)**: Characteristics like baseline loneliness or a tendency to anthropomorphize affect how users write prompts (influencing T) and how readily they form attachments (influencing Y).
    * **Control Method:** While harder to measure directly, we can use **`hashed_ip` + `header`** from your dataset as a proxy for the user. This allows us to potentially control for a user's average behavior across multiple conversations.
* **X3: Conversation Context (Dynamic Confounder)**: Factors like the **`turn` number** and **`timestamp`** (time of day) can influence both the immediate prompt/response (T) and the user's state (Y). A conversation late at night or deep into its turns has different dynamics.
    * **Control Method:** Filter out short conversations. We can include `turn` number as a variable in our matching algorithm alongside the prompt embedding.

* **X4 User's Goal:** This is a powerful confounder. A user with a "socio-emotional goal" will write a prompt that *causes* the LLM to be empathetic (T) and *also* *causes* them to be more open to attachment (Y).
    * **Control Method:** Our plan to use **semantic matching on the user's initial prompt** is the perfect way to control for this. A prompt with a "socio-emotional goal" (e.g., "I'm feeling really down today") is semantically completely different from a "task-focused goal" (e.g., "Write a Python function").

* **X5: LLM Model:** This is also a confounder. A more advanced model (like GPT-4) might be inherently better at being empathetic (affects T) and simultaneously better at building rapport in other ways (affects Y), which would bias your results.

    * **Control Method:** Perform our matching analysis *separately for each model group*.

1.  **Filter:** Create one dataset for `gpt-4` conversations and another for `gpt-3.5-turbo`.
2.  **Match:** Perform your semantic prompt matching *within* the `gpt-4` dataset. Then, do a separate matching *within* the `gpt-3.5-turbo` dataset.
3.  **Calculate:** You will get two ATEs:
    * The ATE of empathy for `gpt-4` users.
    * The ATE of empathy for `gpt-3.5-turbo` users.
    * This is a very robust and clean way to control for the `model` as a confounder. You can then report these two separate ATEs or, if you need a single number, report the average of the two.

---
### 3. Mediating Variables (M): The "How" - Mechanisms 

These variables lie *on* the causal path and explain *how* T causes Y. They represent the psychological process.

* **M1: User's Perceived Understanding**: The empathetic LLM response (T) likely causes the user to *feel* understood and validated (M1), and this *feeling* then leads to the expression of attachment (Y).
* **M2: User's Emotional State Change**: Empathy (T) might work by reducing negative emotions or increasing positive ones (M2), which then results in an attached reply (Y). 

---

```mermaid
graph TD
    subgraph "Confounders (X)"
        X1["X1: User's Initial Prompt (Goal/Emotion)"]
        X2["X2: User's Latent Traits (Loneliness, etc.)"]
        X3["X3: Conversation Context (Turn #, Time)"]
        X4["X4: LLM Model ID (e.g., GPT-4 vs. 3.5)"]
    end
    
    subgraph "Causal Path"
        T["T: LLM Empathy Score"]
        M1["M1: User's Perceived Understanding"]
        M2["M2: User's Emotional Change"]
        Y["Y: User Attachment Score"]
    end

    %%subgraph "Collider (to avoid)"
      %%  C1["C1: User Vote ('Winner')"]
    %%end

    %% Confounder Paths
    X1 --> T
    X1 --> Y
    X2 --> T
    X2 --> Y
    X3 --> T
    X3 --> Y
    X4 --> T
    X4 --> Y

    %% Causal Path (with Mediators)
    T --> M1 --> Y
    T --> M2 --> Y

    %% Collider Path
    %%T --> C1
    %%Y --> C1

    %% Styling
    classDef confounder fill:#fdd,stroke:#b00,stroke-width:2px,color:#b00
    classDef collider fill:#dfd,stroke:#0b0,stroke-width:2px,color:#0b0
    class X1,X2,X3,X4 confounder
    class C1 collider