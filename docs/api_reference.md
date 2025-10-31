# API Reference

## Core Module (`src.core`)

### SWELM

Main class implementing Semantic Weighting for Equitable Language Modeling.

```python
class SWELM(encoder_model, alpha, beta, device)
```

**Parameters:**
- `encoder_model` (str): Name of the multilingual encoder model
- `alpha` (float): Weight for semantic distance component
- `beta` (float): Weight for representation quality component
- `device` (str, optional): Device to run computations on

**Methods:**

#### compute_semantic_weights

```python
compute_semantic_weights(texts, languages, reference_embeddings=None)
```

Compute semantic weights for a batch of texts.

**Parameters:**
- `texts` (List[str]): List of input texts
- `languages` (List[str]): List of language codes
- `reference_embeddings` (torch.Tensor, optional): Reference embeddings

**Returns:** `np.ndarray` - Array of semantic weights

#### reweight_samples

```python
reweight_samples(samples, weights)
```

Apply semantic weights to training samples.

**Parameters:**
- `samples` (List[Dict]): List of training samples
- `weights` (np.ndarray): Computed semantic weights

**Returns:** `List[Dict]` - Reweighted samples

---

## Encoders Module (`src.encoders`)

### MultilingualEncoder

Wrapper for multilingual encoders.

```python
class MultilingualEncoder(model_name, device)
```

**Parameters:**
- `model_name` (str): HuggingFace model identifier
- `device` (str, optional): Device to run the model on

**Methods:**

#### encode

```python
encode(texts, batch_size=32, max_length=512)
```

Encode texts into embeddings.

**Returns:** `torch.Tensor` - Tensor of embeddings

#### compute_similarity

```python
compute_similarity(embeddings1, embeddings2)
```

Compute cosine similarity between embeddings.

**Returns:** `torch.Tensor` - Similarity scores

---

## Adaptive Sampling (`src.adaptive`)

### AdaptiveSampler

Implements adaptive sampling strategies.

```python
class AdaptiveSampler(sampling_strategy, temperature, min_weight)
```

**Parameters:**
- `sampling_strategy` (str): Strategy ('proportional', 'sqrt', 'log')
- `temperature` (float): Temperature for softmax sampling
- `min_weight` (float): Minimum weight threshold

**Methods:**

#### compute_sampling_probabilities

```python
compute_sampling_probabilities(weights)
```

Convert semantic weights to sampling probabilities.

**Returns:** `np.ndarray` - Sampling probabilities

#### sample_batch

```python
sample_batch(data, weights, batch_size, replace=False)
```

Sample a batch according to semantic weights.

**Returns:** `List[Dict]` - Sampled batch

---

## Metrics Module (`src.metrics`)

### Functions

#### compute_exact_match

```python
compute_exact_match(predictions, references)
```

Compute exact match accuracy.

**Returns:** `float` - Exact match score

#### compute_f1_score

```python
compute_f1_score(predictions, references)
```

Compute token-level F1 score.

**Returns:** `float` - F1 score

#### evaluate_performance

```python
evaluate_performance(predictions, references, languages=None, metrics=['exact_match', 'f1'])
```

Evaluate performance across multiple metrics.

**Returns:** `Dict[str, float]` - Dictionary of metric scores

---

## LLM Interface (`models.llm_interface`)

### LLMInterface

Abstract interface for language models.

```python
class LLMInterface(model_name, **kwargs)
```

**Abstract Methods:**

#### generate

```python
generate(prompts, max_length=512, temperature=0.7, **kwargs)
```

Generate responses for a batch of prompts.

**Returns:** `List[str]` - List of generated responses

#### get_logprobs

```python
get_logprobs(prompts, continuations)
```

Compute log probabilities of continuations.

**Returns:** `List[float]` - List of log probabilities

---

## Llama Wrapper (`models.llama_wrapper`)

### Llama3Wrapper

Wrapper for Llama 3 models.

```python
class Llama3Wrapper(model_name, device, load_in_8bit)
```

**Parameters:**
- `model_name` (str): HuggingFace model identifier
- `device` (str, optional): Device to load model on
- `load_in_8bit` (bool): Whether to use 8-bit quantization

Inherits all methods from `LLMInterface`.

---

## Utilities (`src.utils`)

### Functions

#### load_config

```python
load_config(config_path)
```

Load configuration from YAML file.

**Returns:** `Dict[str, Any]` - Configuration dictionary

#### save_results

```python
save_results(results, output_path)
```

Save results to JSON file.

#### set_seed

```python
set_seed(seed=42)
```

Set random seed for reproducibility.

#### setup_logging

```python
setup_logging(log_file=None, level=logging.INFO)
```

Setup logging configuration.
