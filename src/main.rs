use serde::Deserialize;
use std::collections::HashSet;
use std::error::Error;

#[derive(Debug, Deserialize)]
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

fn main() {
    let tokens: Vec<&str> = tokenize("this is a test");
    let records: Vec<Record> = read_training_set().unwrap();
    let vocabulary: HashSet<String> = build_vocabulary(records);
    let vector: Vec<u16> = text_to_vector(tokens, vocabulary);
}
