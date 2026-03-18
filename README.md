# Perceptron-RS

A simple text classifier built from scratch in Rust. It uses a perceptron with softmax to classify SEC financial documents into categories: 10-K, 10-Q, 8-K, 10-K/A, and 10-Q/A.

## How it works

- Reads a CSV file with text and labels
- Builds a vocabulary from the training data
- Trains using stochastic gradient descent (SGD) with cross-entropy loss
- Evaluates on test data and runs manual predictions

## Run it

```bash
cargo run
```

You'll need a `financial_report_text_training_expanded.csv` file in the project root with `text` and `label` columns.

## What it does

The model converts text to a bag-of-words vector, runs a forward pass to get class probabilities via softmax, then backpropagates to update weights.