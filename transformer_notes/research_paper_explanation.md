
---
# Sequence to Sequence Learning with Neural Networks

# 1. Why RNNs Are Used for Sequences

A **Recurrent Neural Network (RNN)** is an extension of a normal neural network that can process **sequential data**.

Examples of sequences:
* sentences (words in order)
* speech signals
* time series
* DNA sequences

Unlike feed-forward networks, an RNN **remembers previous inputs**.

Example Sentence:

```
I love machine learning
```

The model processes it **one word at a time**.

---

# 2. Mathematical Equation of an RNN

The paper defines two equations.

### Hidden state update

![Equation - 1](\transformer_notes\assets\equation-1.png)



So the RNN **remembers past information**.

---

### Output equation

![Equation - 1](..\transformer_notes\assets\output-eq-1.png)

Meaning:

The output at time **t** is computed from the hidden state.

---

# 3. When Standard RNNs Work Well

RNNs work easily when:

* **input length = output length**
* **alignment is known**

Example: POS tagging

```
Input : I   love   apples
Output: PRON VERB  NOUN
```

Each input word maps directly to one output.

---

# 4. Why Translation Is Hard

Machine translation has **different sequence lengths**.

Example:

English:

```
I love apples
```

French:

```
J'aime les pommes
```

Lengths differ.

Also **word order may change**.

Example:

English:

```
I eat apples
```

German:

```
Ich esse Äpfel
```

So **simple RNN alignment does not work**.

---

# 5. Solution: Encoder–Decoder Architecture

The simplest strategy is:

- Encode the input sentence
- Decode the output sentence

Architecture:

```
Input sentence
     ↓
Encoder RNN
     ↓
Vector representation
     ↓
Decoder RNN
     ↓
Output sentence
```

Example:

```
English: I love apples
           ↓
Vector: [meaning representation]
           ↓
French: J'aime les pommes
```

---

# 6. Problem with Simple RNN Encoder

This method causes **long-term dependency problems**.

Example sentence:

```
The boy who was playing in the park yesterday bought ice cream
```

Important information from the beginning must be remembered **until the end**.

Standard RNNs struggle because of:

* **vanishing gradients**
* **exploding gradients**

Training becomes difficult.

---

# 7. Why LSTM Is Used

The paper uses **Long Short‑Term Memory (LSTM)**.

LSTMs solve the **long dependency problem**.

They use special components:

* input gate
* forget gate
* output gate
* memory cell

These allow the network to **store information for long periods**.

Example:

Sentence:

```
The book that I borrowed yesterday was interesting
```

The model remembers the subject **"book"** even after many words.

---

# 8. What the Model Tries to Learn

The model tries to learn the probability:
![Equation - 1](..\transformer_notes\assets\equation-2.png)


Important:

```
T ≠ T'
```

Input and output lengths may differ.

---

# 9. Encoder Step

The encoder LSTM reads the entire sentence.

Example:

```
Input: A B C <EOS>
```

Processing:

```
h1 → after A
h2 → after B
h3 → after C
h4 → after <EOS>
```

The **last hidden state** becomes the sentence representation:

```
v = h4
```

This vector **v** represents the meaning of the whole sentence.

---

# 10. Decoder Step

The decoder generates the output sentence **one word at a time**.

Probability formula:
![Equation - 1](..\transformer_notes\assets\equation-3.png)


Example generation:

```
Step 1 → W
Step 2 → X
Step 3 → Y
Step 4 → Z
Step 5 → <EOS>
```

So the model predicts:

```
p(W | v)
p(X | v, W)
p(Y | v, W, X)
...
```

---

# 11. Softmax Output Layer

Each word probability is computed using **softmax**.

Softmax converts scores into probabilities.

Example vocabulary:

```
["Je", "Tu", "Il", "Nous"]
```

Softmax output:

```
Je   → 0.72
Tu   → 0.10
Il   → 0.12
Nous → 0.06
```

The model selects the **highest probability word**.

---

# 12. End-of-Sentence Token

They add a special token:

```
<EOS>
```

Example:

```
Input: I love apples <EOS>
Output: J'aime les pommes <EOS>
```

This tells the model **when to stop generating words**.

---

# 13. Using Two Separate LSTMs

Instead of one network, they used:

```
Encoder LSTM
Decoder LSTM
```

Reason:

* increases model capacity
* allows training on multiple language pairs
* improves flexibility

---

# 14. Deep LSTM (Multiple Layers)

They discovered that **deep networks work better**.

So they used:

```
4-layer LSTM
```

Architecture:

```
Layer 1
Layer 2
Layer 3
Layer 4
```

This increases the model's ability to learn **complex language patterns**.

---

# 15. Key Trick: Reversing the Input Sentence

One of the **most important contributions** of the paper.

Instead of training with:

```
Input : a b c
Output: α β γ
```

They reverse the input:

```
Input : c b a
Output: α β γ
```

Example:

Normal:

```
English: I love apples
French : J'aime les pommes
```

Training input:

```
apples love I
```

---

# 16. Why Reversing Helps

This reduces **distance between related words**.

Normal case:

```
I love apples
↓
J'aime les pommes
```

Dependencies are long.

After reversing:

```
apples love I
↓
J'aime les pommes
```

Now the first words are closer to each other during training.

This makes it easier for **Stochastic Gradient Descent (SGD)** to learn the relationship.

Result:

* faster training
* better accuracy
* better handling of long sentences

---

# 17. Final Architecture

Complete model:

```
Input sentence (reversed)
        ↓
Encoder LSTM (4 layers)
        ↓
Vector representation (v)
        ↓
Decoder LSTM (4 layers)
        ↓
Softmax
        ↓
Output sentence
```

---

# 18. Why This Model Is Important

This architecture introduced the **seq2seq paradigm**, which became the foundation for:

* neural machine translation
* chatbots
* speech recognition
* summarization
* modern AI assistants

Later improvements included:

* **Attention Mechanism**
* **Transformer**

---
# PART - 2

# 1. What They Did in the Experiments

The researchers tested their LSTM translation model on the **WMT 2014 dataset** (English → French translation).

They used their model in **two ways**:

### 1️) Direct Translation

The LSTM directly translates English sentences to French.

Example:

```
English: I love apples
↓
LSTM model
↓
French: J'aime les pommes
```

---

### 2️) Rescoring SMT Outputs

They also used the LSTM to **improve translations produced by a traditional system** called **Statistical Machine Translation (SMT)**.

Process:

1. SMT generates **1000 possible translations**.
2. The LSTM evaluates each translation.
3. The best translation is selected.

This improves translation quality.

---

# 2. Dataset Details

They trained their model using:

* **12 million sentence pairs**
* **304 million English words**
* **348 million French words**

Example training pair:

```
English: I love machine learning
French : J'aime l'apprentissage automatique
```

The dataset was chosen because it had:

* clean data
* tokenized sentences
* SMT baseline outputs

---

# 3. Vocabulary Size

Neural networks require **fixed vocabularies**.

So they limited the vocabulary to:

| Language | Vocabulary Size |
| -------- | --------------- |
| English  | 160,000 words   |
| French   | 80,000 words    |

If a word was not in the vocabulary, it was replaced by:

```
UNK
```

Example:

```
I love cryptocurrency
↓
I love UNK
```

This is called an **unknown token**.

---

# 4. How the Model Was Trained

They trained the model to **maximize the probability of the correct translation**.

Goal:

```
Given source sentence S
Predict correct translation T
```

Mathematically:

![Equation - 1](..\transformer_notes\assets\equation-4.png)

Meaning:

The model tries to make the **correct translation very probable**.

Example:

```
Input : I like coffee
Correct output : J'aime le café
```

The model adjusts its weights so that:

```
P("J'aime le café" | "I like coffee") → high
```

---

# 5. How the Model Generates Translations

After training, the model must find the **most likely translation**.

Goal:

![Equation - 1](..\transformer_notes\assets\equation-5.png)

Meaning:

Find the translation **T** with the highest probability.

But checking **all possible sentences** is impossible.

So they use **beam search**.

---

# 6. Beam Search (Decoding Method)

Beam search generates translations **step-by-step**.

Example translation generation:

Input:

```
I love apples
```

Step 1 possibilities:

```
Je
J'
Il
```

Step 2 possibilities:

```
Je aime
J'aime
Je adore
```

Step 3 possibilities:

```
J'aime les
Je adore les
```

Beam search keeps only **B best partial sentences**.

Example:

Beam size **B = 2**

Keep only:

```
J'aime
Je adore
```

Other options are discarded.

This process continues until the **<EOS> token** appears.

---

### End-of-sentence token

```
<EOS>
```

Example:

```
J'aime les pommes <EOS>
```

Once `<EOS>` appears, the translation is **complete**.

---

### Interesting observation

The model worked well even with:

```
Beam size = 1
```

And beam size **2 already gives most benefits**.

This means the model predictions are **very confident**.

---

# 7. Rescoring SMT Outputs

The baseline SMT system generates:

```
1000 translation candidates
```

Example:

```
1. Je aime les pommes
2. J'aime les pommes
3. Je adore les pommes
...
1000 candidates
```

The LSTM computes the probability of each sentence.

Final score =

```
(SMT score + LSTM score) / 2
```

The translation with the **highest combined score wins**.

This improves the overall translation quality.

---

# 8. Reversing Source Sentences

One of the **most important discoveries**.

Instead of training with:

```
Input : I love apples
Output: J'aime les pommes
```

They reverse the input:

```
Input : apples love I
Output: J'aime les pommes
```

---

# 9. Why Reversing Helps

Normally, corresponding words are **far apart**.

Example:

```
I love apples
↓
J'aime les pommes
```

Word alignment:

```
I      → J'
love   → aime
apples → pommes
```

But in the RNN sequence, the distance is large.

This creates **long time lag**.

---

### After reversing

```
apples love I
↓
J'aime les pommes
```

Now some corresponding words appear **closer together during training**.

This reduces **minimal time lag**.

---

# 10. What Is Minimal Time Lag?

It is the **distance between related input and output words**.

Large lag example:

```
Input word → many steps later → output word
```

This makes learning difficult because gradients must travel **many time steps**.

---

# 11. Why Reversing Helps Learning

Reversing sentences:

* creates **shorter dependencies**
* helps **backpropagation**
* improves training

Because gradients can flow more easily.

---

# 12. Experimental Results of Reversing

Reversing input sentences improved performance significantly.

| Metric     | Before reversing | After reversing |
| ---------- | ---------------- | --------------- |
| Perplexity | 5.8              | **4.7**         |
| BLEU score | 25.9             | **30.6**        |

Explanation:

* **Lower perplexity = better language modeling**
* **Higher BLEU = better translation**

---

# 13. Unexpected Benefit

Researchers initially thought reversing would help **only early words** in translation.

But they discovered something surprising:

The model also became **better at translating long sentences**.

This means:

* LSTM memory usage improved
* the model learned dependencies better

---

# 14. Final Key Ideas of This Section

This experiment section shows:

- The model was trained on **12M sentence pairs**
- Vocabulary was limited to **160k English and 80k French words**
- Translations were generated using **beam search**
- LSTM was also used to **rescore SMT translations**
- **Reversing source sentences greatly improved performance**

---
**In simple terms:**

The researchers trained a large LSTM translation model and discovered that:

* it works well for translation
* beam search helps generate good sentences
* using LSTM to rescore SMT improves results
* **reversing the input sentence dramatically improves learning**

---

