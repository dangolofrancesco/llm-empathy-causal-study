# llm-empathy-causal-study
With this project we aim to determine if a more empathetic chatbot response causally increases a user’s immediate emotional attachment.

# Research Project Causal Model

This diagram outlines the causal relationships, confounders, moderators, and mediators in our study.

## Causal Model (DAG)

This diagram outlines the potential causal relationships in the study, including confounders, moderators, and mediators.

```mermaid
graph TD
    subgraph "Confounders (X)"
        X1["X1: User's Initial Prompt (Topic/Emotion)"]
        X2["X2: User's Latent Traits (e.g., Loneliness, Anthropomorphism)"]
        X3["X3: Conversation Context (e.g., Time of Day, Turn Number)"]
    end
    
    subgraph "Moderators (Z)"
        Z1["Z1: LLM Model ID (e.g., GPT-4o vs. Llama 3)"]
        Z2["Z2: User's Goal (Task-focused vs. Socio-emotional)"]
    end

    subgraph "Causal Path"
        T[T: LLM Empathy Score]
        M1["M1: User's Perceived Understanding"]
        M2["M2: User's Emotional Change (The 'Difference')"]
        Y[Y: User Attachment Score]
    end

    subgraph "Collider (to avoid)"
        C1["C1: User Vote ('Winner')"]
    end

    %% Confounder Paths
    X1 --> T
    X1 --> Y
    X2 --> T
    X2 --> Y
    X3 --> T
    X3 --> Y

    %% Causal Path (with Mediators)
    T --> M1 --> Y
    T --> M2 --> Y

    %% Moderator Paths (showing they affect the T->Y relationship)
    Z1 -- affects --> T
    Z1 -- affects --> Y
    Z2 -- affects --> T
    Z2 -- affects --> Y


    %% Collider Path
    T --> C1
    Y --> C1

    %% Styling
    classDef confounder fill:#fdd,stroke:#b00,stroke-width:2px
    classDef moderator fill:#ddf,stroke:#00b,stroke-width:2px
    classDef collider fill:#dfd,stroke:#0b0,stroke-width:2px
    class X1,X2,X3 confounder
    class Z1,Z2 moderator
    class C1 collider