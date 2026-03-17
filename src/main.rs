use serde::Deserialize;
use std::collections::HashSet;
use std::error::Error;

#[derive(Debug, Deserialize, Clone)]
struct Record {
    text: String,
    label: String,
}

fn read_training_set() -> Result<Vec<Record>, Box<dyn Error>> {
    let mut data = csv::Reader::from_path("financial_report_text_training_expanded.csv")?;

    let mut records: Vec<Record> = Vec::new();
    for result in data.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    Ok(records)
}

fn tokenize(text: &str) -> Vec<&str> {
    let v: Vec<&str> = text.split(" ").collect();
    v
}

fn build_vocabulary(records: Vec<Record>) -> HashSet<String> {
    let mut vocabulary: HashSet<String> = HashSet::new();
    for record in records {
        for word in tokenize(&record.text) {
            vocabulary.insert(word.to_string());
        }
    }

    vocabulary
}

fn text_to_vector(tokens: Vec<&str>, vocabulary: HashSet<String>) -> Vec<u16> {
    let mut vector: Vec<u16> = vec![0u16; vocabulary.len()];

    for token in tokens {
        if let Some(index) = vocabulary.iter().position(|v| v == token) {
            vector[index] += 1;
        }
    }

    vector
}

fn encode_labels(records: Vec<Record>) -> Vec<u8> {
    let mut labels: Vec<u8> = Vec::new();

    for record in records {
        let encoded: u8 = match record.label.as_str() {
            "10-K" => 0,
            "10-Q" => 1,
            "8-K" => 2,
            "10-K/A" => 3,
            "10-Q/A" => 4,
            _ => 255,
        };

        labels.push(encoded);
    }

    labels
}

fn init_weight(input_dimensions: usize, output_dimensions: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let w = vec![vec![0.01; output_dimensions]; input_dimensions];

    let b = vec![0.0; output_dimensions];

    (w, b)
}

fn forward_pass(vector: Vec<u16>, w: Vec<Vec<f32>>, b: Vec<f32>) -> Vec<f32> {
    let mut logits: Vec<f32> = b.clone();
    for (i, &token_count) in vector.iter().enumerate() {
        for (j, &weight) in w[i].iter().enumerate() {
            logits[j] += token_count as f32 * weight;
        }
    }

    logits
}

fn softmax(logits: Vec<f32>) -> Vec<f32> {
    let exps: Vec<f32> = logits.iter().map(|&x| x.exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

/* TODO
Compute loss — cross-entropy between your predicted probabilities and the true label
Backward pass — calculate gradients of the loss with respect to W and b
Update weights — nudge W and b in the direction that reduces loss (SGD)
Loop — repeat steps 3–6 over your training data for several epochs
Evaluate — run on held-out test documents, check accuracy and a confusion matrix
*/

fn main() {
    let tokens: Vec<&str> = tokenize("this is a test");
    let records: Vec<Record> = read_training_set().unwrap();
    let vocabulary: HashSet<String> = build_vocabulary(records.clone());
    let labels: Vec<u8> = encode_labels(records);
    let vector: Vec<u16> = text_to_vector(tokens, vocabulary);
    let (w, b) = init_weight(vector.len(), labels.len());
    let logits: Vec<f32> = forward_pass(vector, w, b);
    let probabilities = softmax(logits);
}
