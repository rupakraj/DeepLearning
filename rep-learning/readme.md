# Representation Learning

Turn every customer in a CRM table into a vector with an Encoder–Decoder, then search customers with plain English.


## The problem

A database filter only answers questions you can write as exact rules on known columns. We want to:

* search with plain English: *"unhappy healthcare customers"*, *"high value enterprise accounts likely to churn"*
* find customers **similar** to a given one (there is no "like" in SQL)
* match a **new lead** where only a few fields are known

The idea: learn a short vector (32 numbers) per customer, so that similar customers get similar vectors.
Search is then one dot product between a query vector and every customer vector.

## Data

`data/crm.csv` is a synthetic CRM table: 3,000 customers, 32 columns. The columns are correlated (unhappy customers
have a higher churn risk, and their notes sound unhappy too), so there is a real signal to learn.

| Kind | Columns |
| --- | --- |
| Category (12) | city, state, country, timezone, industry, job_title, plan_tier, lifecycle_stage, acquisition_source, preferred_channel, last_ticket_category, newsletter_opt_in |
| Number (7) | total_orders, monthly_recurring_revenue, lifetime_value, avg_order_value, open_tickets, satisfaction_score, churn_risk |
| Date (3) | signup_date, last_purchase_date, last_contact_date → days since |
| Text (3) | notes, interests, last_ticket_subject |
| Not used | customer_id, names, email, phone, date_of_birth, company |

Notebook 1 turns each kind into numbers.
- Categories become ids, numbers are scaled
- Dates become "days since".
- Text becomes hashed word ids.
- Missing values are **marked**, not guessed.
- It also gives each customer a few plain-English **tags** (e.g. `industry=Healthcare`, `unhappy`), which are used to make practice search queries.

To regenerate the data:

```bash
python scripts/make_crm.py 3000 data/crm.csv
```

## Model: Encoder–Decoder

```
                 ┌─────────┐                ┌─────────┐
customer row ──► │ Encoder │ ──► vector ──► │ Decoder │ ──► customer row (rebuilt)
 (26 fields)     └─────────┘  (32 numbers)  └─────────┘
```

The 26 fields are the 25 columns above plus an extra text field, `description`, which is empty for real customers.
A **search query is a customer where only the description is known**. So customers and queries go through the same
Encoder and land in the same space.

### Encoder: 26 fields → 32 numbers

```
 city  industry … plan   orders  revenue …  notes  interests  ticket  description
   │      │        │       │       │          │        │         │         │
  64     64       64      64      64         64       64        64        64    ← a vector per field
   └────────────── side by side: 26 × 64 = 1664 numbers (missing field = zeros) ──────┘
                                          │
                                 MLP: 1664 → 256 → 32
                                          │
                              L2-normalize → customer vector
```

| Step | Layer | Output shape |
| --- | --- | --- |
| Category fields (12) | one `nn.Embedding(vocab + 1, 64)` per column (id 0 = unknown) | B × 12 × 64 |
| Number fields (10) | `value × w + b`, with learned `w`, `b` of size 64 per column | B × 10 × 64 |
| Text fields (4) | shared `nn.EmbeddingBag(50000, 64, mode="mean")` over hashed word ids | B × 4 × 64 |
| Mask | concatenate, set missing fields to 0 | B × 26 × 64 |
| MLP | flatten → `Linear(1664, 256)` → `ReLU` → `Linear(256, 32)` | B × 32 |
| Normalize | scale to length 1, so similarity = dot product | B × 32 |

### Decoder: 32 numbers → 22 fields

| Step | Layer | Output shape |
| --- | --- | --- |
| MLP | `Linear(32, 256)` → `ReLU` | B × 256 |
| Category heads (12) | one `Linear(256, vocab + 1)` per column, a classifier | B × (vocab + 1), for each column |
| Number head | `Linear(256, 10)`, a regression | B × 10 |

Text is used only as input and is not rebuilt.

### Parameter size

| Part | Layer | Calculation | Parameters |
| --- | --- | --- | ---: |
| Encoder | Category embeddings (12) | (sum of vocab + 1 = 117) × 64 | 7,488 |
| | Number `w` and `b` | 2 × 10 × 64 | 1,280 |
| | Word embeddings | 50,000 × 64 | 3,200,000 |
| | `Linear(1664, 256)` | 1664 × 256 + 256 | 426,240 |
| | `Linear(256, 32)` | 256 × 32 + 32 | 8,224 |
| | **Encoder total** | | **3,643,232** |
| Decoder | `Linear(32, 256)` | 32 × 256 + 256 | 8,448 |
| | Category heads (12) | 117 × 256 + 117 | 30,069 |
| | `Linear(256, 10)` | 256 × 10 + 10 | 2,570 |
| | **Decoder total** | | **41,087** |
| | **Model total** | | **3,684,319** (~3.6M) |

88% of the Encoder is the word-embedding table. The Decoder is small because it only has to rebuild 22 fields from 32 numbers.

## Training

Every batch of training customers plays two games, and the two losses are added.

### Game 1: fill in the blanks (denoising autoencoder)

If the Encoder sees every field, rebuilding them is too easy. So each field is **hidden with probability 0.5**
(on top of the fields that are already missing), and the Decoder must rebuild **all** of them:

$$
\tilde{x} = \text{hide}(x), \qquad z = \text{Enc}(\tilde{x}), \qquad (\hat{y}_1, \dots, \hat{y}_{12}, \hat{n}) = \text{Dec}(z)
$$

$$
\mathcal{L}_{\text{rebuild}} = \underbrace{\frac{1}{12} \sum_{c=1}^{12} \text{CE}(\hat{y}_c, y_c)}_{\text{categories}}
\; + \; \underbrace{\frac{1}{|K|} \sum_{(i,j) \in K} (\hat{n}_{ij} - n_{ij})^2}_{\text{numbers}}
$$

* $y_c$ is the true id of category column $c$. Rows where it is unknown (id 0) are skipped.
* $K$ is the set of (customer, number column) pairs whose real value is known.

The vector learns to summarize the customer and how the fields relate (angry notes go with a low satisfaction score).

### Game 2: match the description (contrastive)

Game 1 never teaches that *"unhappy"* means a low `satisfaction_score`, since that word is in no column. So for each
customer $i$ in the batch, a description $d_i$ is made from 1–3 of its tags (e.g. *"frustrated healthcare company"*).
Each description must be closest to **its own** customer among the $B$ customers in the batch:

$$
q_i = \text{Enc}(d_i), \qquad z_j = \text{Enc}(x_j), \qquad s_{ij} = \frac{q_i \cdot z_j}{\tau}
$$

$$
\mathcal{L}_{\text{match}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s_{ii}}}{\sum_{j=1}^{B} e^{s_{ij}}}
$$

This is a cross-entropy where the right "class" for query $i$ is customer $i$. The temperature $\tau = 0.1$ makes
the matching sharper. Both $q$ and $z$ come from the **same** Encoder, so queries and customers share one space.

### Total objective

$$
\mathcal{L} = \mathcal{L}_{\text{rebuild}} + \mathcal{L}_{\text{match}}
$$

| Setting | Value |
| --- | --- |
| Training data | 80% of customers (the other 20% are kept for testing) |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 256 |
| Epochs | 200 on GPU, 40 on CPU |

After training, the weights are saved to `outputs/model.pt` and loaded back for searching.


## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For GPU, Jupyter kernel, data and Google Colab, see [env-setup.md](env-setup.md).


## Notebooks

Run them in order.

| SN | Notebook | What it does |
| --- | --- | --- |
| 1 | [Dataset preparation](01_dataset_preparation.ipynb) | Turn categories, numbers, dates and text into model input |
| 2 | [Model and training](02_model_and_training.ipynb) | Train the Encoder–Decoder, save it, and search with it |
| 3 | [Merged (Dataset + Model)](building_embedding.ipynb) | Merge of (1) and (2) notebook |