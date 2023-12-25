### Description
Rust Implementation of the [Weng-Lin Rating](https://www.csie.ntu.edu.tw/~cjlin/papers/online_ranking/online_journal.pdf).

This library is based on the [openskill.js](https://github.com/philihp/openskill.js) library

### Installation
```
cargo install openskill
```

### Usage
```rust
use openskill::prelude::*;

fn main() -> Result<(), OpenSkillError> {
    let team1 = vec![Rating::default()]; 
    let team2 = vec![Rating::new(35.0, 7.0)];

    let teams = vec![team1.clone(), team2.clone()];

    let env = Env::default(); // == EnvBuilder::default().model(ModelKind::PlackettLuce).build()
    let draw_prob = env.predict_draw(&teams)?;
    let win_rate = env.predict_win(&teams)?;
    println!("teams: {teams:?}");
    println!("draw probability: {draw_prob}, win rate: {win_rate:?}");

    let new_rate = env.rate(&GameResult::new(teams.clone(), vec![1, 2]))?;
    println!("before: {teams:?}\n after: {new_rate:?}");
    Ok(())
}
```