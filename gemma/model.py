"""

Gemini, developed by Google DeepMind, utilizes a Mixture-of-Experts (MoE) architecture,
which distinguishes it from the traditional Transformer architecture used by models like GPT-4 from OpenAI

differences to original transformer

Multi-Query Attention
- Notably, the 7B model uses multi-head attention while the 2B checkpoints use multi-query attention (with 𝑛𝑢𝑚_𝑘𝑣_ℎ𝑒𝑎𝑑𝑠 = 1), based on ablations
that showed that multi-query attention works well at small scales 

RoPE Embeddings
-  Rather than using absolute positional embeddings, we use rotary positional embeddings in each layer; we also
share embeddings across our inputs and outputs to reduce model size.

GeGLU Activations
- The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function.

RMSNorm
- We normalize the input of each transformer sub-layer, the attention layer and the feedforward layer, with RMSNorm


"""