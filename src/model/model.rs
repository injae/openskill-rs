use crate::rating::Rating;

pub trait Model {
    fn rate(&self, teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Vec<Vec<Rating>>;
}
