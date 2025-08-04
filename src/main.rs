use std::collections::{BTreeSet, HashMap};

fn main(){
    let text = include_str!("input.txt");
    println!("Text chars len: {}", text.chars().count());

    let chars = text.chars().collect::<BTreeSet<char>>();

    let stoi = chars.
        iter().
        enumerate().
        map(|(i, ch)|(*ch, i as i32)).
        collect::<HashMap<char, i32>>();

    println!("{stoi:?}");
}