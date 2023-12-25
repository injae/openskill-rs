use crate::constant::{DEFAULT_MU, DEFAULT_SIGMA};
use std::{fmt::Display, iter::zip};

#[derive(Debug, Clone)]
pub struct Rating {
    pub mu: f64,
    pub sigma: f64,
}

impl Display for Rating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(mu: {}, sigma: {})", self.mu, self.sigma)
    }
}

impl Default for Rating {
    fn default() -> Self {
        Self {
            mu: DEFAULT_MU,
            sigma: DEFAULT_SIGMA,
        }
    }
}

impl Rating {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }
}

#[derive(Debug, Clone)]
pub struct TeamRating {
    pub members: Vec<Rating>,
    pub mu: f64,
    pub sigma_sq: f64,
    pub rank: usize,
}

impl TeamRating {
    pub fn new(members: Vec<Rating>, rank: usize) -> Self {
        let mu = members.iter().map(|m| m.mu).sum::<f64>();
        let sigma_sq = members.iter().map(|m| m.sigma.powi(2)).sum::<f64>();
        Self {
            members,
            mu,
            sigma_sq,
            rank,
        }
    }

    pub fn team_size(&self) -> usize {
        self.members.len()
    }
}

pub(crate) fn to_team_ratings(
    teams: &Vec<Vec<Rating>>,
    ranks: Option<Vec<usize>>,
) -> Vec<TeamRating> {
    let ranks = ranks.unwrap_or_else(|| (0..teams.len()).collect());
    zip(teams.iter(), ranks.iter())
        .map(|(team, rank)| TeamRating::new(team.clone(), *rank))
        .collect()
}

pub(crate) fn calc_total_player(team_ratings: &Vec<TeamRating>) -> usize {
    team_ratings.iter().map(|it| it.team_size()).sum::<usize>()
}

pub struct GameResult {
    pub teams: Vec<Vec<Rating>>,
    pub ranks: Vec<usize>,
}

impl GameResult {
    pub fn new(teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Self {
        Self { teams, ranks }
    }
}

pub type GammaFunc = fn(f64, f64, team: &TeamRating) -> f64;

pub fn default_gamma(c: f64, _: f64, team: &TeamRating) -> f64 {
    team.sigma_sq.sqrt() / c
}

pub type OrdinalFunc = fn(&Rating, f64) -> f64;

pub fn default_ordinal(rating: &Rating, z: f64) -> f64 {
    rating.mu - (z * rating.sigma)
}

pub fn ladder_pairs<T>(ranks: &Vec<T>) -> Vec<Vec<T>>
where
    T: Clone,
{
    let o_ranks: Vec<Option<T>> = ranks.clone().into_iter().map(Some).collect();
    let left: Vec<_> = [None]
        .iter()
        .cloned()
        .chain(o_ranks[..o_ranks.len() - 1].iter().cloned())
        .collect();
    let right = vec![&o_ranks[1..], &[None]].concat();
    return zip(left, right)
        .map(|(l, r)| match (l, r) {
            (Some(l), Some(r)) => vec![l, r],
            (Some(l), None) => vec![l],
            (None, Some(r)) => vec![r],
            _ => vec![],
        })
        .collect();
}
