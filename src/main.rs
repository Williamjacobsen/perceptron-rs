use serde::Deserialize;
use std::collections::HashSet;
use std::error::Error;

#[derive(Debug, Deserialize, Clone)]
struct Record {
    text: String,
    label: String,
}

fn read_records() -> Result<Vec<Record>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path("financial_report_text_training_expanded.csv")?;
    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: Record = result?;
        records.push(record);
    }
    Ok(records)
}

fn train_test_split(records: &[Record], train_ratio: f32) -> (&[Record], &[Record]) {
    let split_index = (records.len() as f32 * train_ratio) as usize;
    records.split_at(split_index)
}

fn build_vocabulary(records: &[Record]) -> Vec<String> {
    let mut vocabulary_set = HashSet::new();

    for record in records {
        for word in record.text.split_whitespace() {
            vocabulary_set.insert(word.to_string());
        }
    }

    let mut vocabulary = vocabulary_set.into_iter().collect::<Vec<String>>();
    vocabulary.sort();
    vocabulary
}

fn encode_label(label_string: &str) -> u8 {
    match label_string {
        "10-K" => 0,
        "10-Q" => 1,
        "8-K" => 2,
        "10-K/A" => 3,
        "10-Q/A" => 4,
        _ => 255,
    }
}

fn decode_label(label_index: u8) -> String {
    match label_index {
        0 => "10-K".to_string(),
        1 => "10-Q".to_string(),
        2 => "8-K".to_string(),
        3 => "10-K/A".to_string(),
        4 => "10-Q/A".to_string(),
        _ => "unknown".to_string(),
    }
}

fn encode_labels(records: &[Record]) -> Vec<u8> {
    let mut labels = Vec::new();
    for record in records {
        labels.push(encode_label(&record.label));
    }
    labels
}

fn text_to_vector(text: &str, vocabulary: &[String]) -> Vec<u16> {
    let mut vector = vec![0u16; vocabulary.len()];

    for word in text.split_whitespace() {
        for (index, vocab_word) in vocabulary.iter().enumerate() {
            if word == vocab_word {
                vector[index] += 1;
                break;
            }
        }
    }

    vector
}

fn init_weights(vocabulary_size: usize, num_classes: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let weights = vec![vec![0.01; num_classes]; vocabulary_size];
    let biases = vec![0.0; num_classes];
    (weights, biases)
}

fn forward_pass(feature_vector: &[u16], weights: &[Vec<f32>], biases: &[f32]) -> Vec<f32> {
    let mut logits = biases.to_vec();

    for feature_index in 0..feature_vector.len() {
        let feature_count = feature_vector[feature_index];
        for class_index in 0..weights[feature_index].len() {
            let weight = weights[feature_index][class_index];
            logits[class_index] += feature_count as f32 * weight;
        }
    }

    logits
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut exp_values = Vec::new();
    for logit in logits {
        exp_values.push(logit.exp());
    }

    let sum: f32 = exp_values.iter().sum();
    let probabilities: Vec<f32> = exp_values.iter().map(|x| x / sum).collect();
    probabilities
}

fn backward_pass(
    probabilities: &[f32],
    true_label: u8,
    feature_vector: &[u16],
    weights: &mut [Vec<f32>],
    biases: &mut [f32],
) {
    let learning_rate = 0.01;

    let mut gradients = probabilities.to_vec();
    gradients[true_label as usize] -= 1.0;

    for feature_index in 0..feature_vector.len() {
        let feature_count = feature_vector[feature_index];
        for class_index in 0..gradients.len() {
            let gradient = gradients[class_index];
            weights[feature_index][class_index] -= learning_rate * gradient * feature_count as f32;
        }
    }

    for class_index in 0..gradients.len() {
        biases[class_index] -= learning_rate * gradients[class_index];
    }
}

fn train(
    records: &[Record],
    vocabulary: &[String],
    labels: &[u8],
    epochs: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let (mut weights, mut biases) = init_weights(vocabulary.len(), 5);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for record_index in 0..records.len() {
            let feature_vector = text_to_vector(&records[record_index].text, vocabulary);
            let logits = forward_pass(&feature_vector, &weights, &biases);
            let probabilities = softmax(&logits);

            let loss = -probabilities[labels[record_index] as usize].ln();
            total_loss += loss;

            backward_pass(
                &probabilities,
                labels[record_index],
                &feature_vector,
                &mut weights,
                &mut biases,
            );
        }

        let avg_loss = total_loss / records.len() as f32;
        println!("epoch {} loss: {:.4}", epoch + 1, avg_loss);
    }

    (weights, biases)
}

fn evaluate(
    records: &[Record],
    vocabulary: &[String],
    labels: &[u8],
    weights: &[Vec<f32>],
    biases: &[f32],
) -> f32 {
    let mut correct_predictions = 0;

    for record_index in 0..records.len() {
        let feature_vector = text_to_vector(&records[record_index].text, vocabulary);
        let logits = forward_pass(&feature_vector, weights, biases);
        let probabilities = softmax(&logits);

        let mut predicted_label = 0u8;
        for i in 1..probabilities.len() {
            if probabilities[i] > probabilities[predicted_label as usize] {
                predicted_label = i as u8;
            }
        }

        if predicted_label == labels[record_index] {
            correct_predictions += 1;
        }
    }

    correct_predictions as f32 / records.len() as f32
}

fn test_manual(text: &str, vocabulary: &[String], weights: &[Vec<f32>], biases: &[f32]) -> String {
    let feature_vector = text_to_vector(text, vocabulary);
    let logits = forward_pass(&feature_vector, weights, biases);
    let probabilities = softmax(&logits);

    let mut predicted_label = 0u8;
    for i in 1..probabilities.len() {
        if probabilities[i] > probabilities[predicted_label as usize] {
            predicted_label = i as u8;
        }
    }

    decode_label(predicted_label)
}

fn main() {
    let records = read_records().expect("Failed to read records");
    println!("Total samples: {}", records.len());

    let (training_set, test_set) = train_test_split(&records, 0.8);
    println!("Training samples: {}", training_set.len());
    println!("Test samples: {}", test_set.len());

    let vocabulary = build_vocabulary(training_set);
    println!("Vocabulary size: {}", vocabulary.len());

    let training_labels = encode_labels(training_set);
    let test_labels = encode_labels(test_set);

    let (weights, biases) = train(training_set, &vocabulary, &training_labels, 10);

    let training_accuracy = evaluate(
        training_set,
        &vocabulary,
        &training_labels,
        &weights,
        &biases,
    );
    let test_accuracy = evaluate(test_set, &vocabulary, &test_labels, &weights, &biases);

    println!("Train accuracy: {:.2}%", training_accuracy * 100.0);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);

    let manual_text = "This report provides a comprehensive overview of the company's annual financial performance including detailed statements of operations and balance sheets";
    let manual_prediction = test_manual(manual_text, &vocabulary, &weights, &biases);
    println!("Manual test: {}", manual_prediction);

    let manual_text2 = "This quarterly report presents unaudited financial statements for the fiscal quarter ending March 31";
    let manual_prediction2 = test_manual(manual_text2, &vocabulary, &weights, &biases);
    println!("Manual test 2: {}", manual_prediction2);
}
