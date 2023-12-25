use crate::model::model::Model;
use crate::rating::{to_team_ratings, GammaFunc, Rating, TeamRating};
use crate::utils::update_team_rating;

pub struct PlackettLuce {
    beta_sq: f64,
    gamma: GammaFunc,
    kappa: f64,
}

impl PlackettLuce {
    pub fn new(beta: f64, kappa: f64, gamma: GammaFunc) -> Self {
        Self {
            beta_sq: beta.powi(2),
            gamma,
            kappa,
        }
    }
}

impl Model for PlackettLuce {
    fn rate(&self, teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Vec<Vec<Rating>> {
        let team_ratings = to_team_ratings(&teams, Some(ranks));
        let c = calc_c(&team_ratings, self.beta_sq);
        let sum_q = calc_sum_q(&team_ratings, c);
        let a = calc_a(&team_ratings);

        let num_teams = team_ratings.len() as f64;
        team_ratings
            .iter()
            .enumerate()
            .map(|(i, team_i)| {
                let (mut omega, mut delta) = (0.0, 0.0);
                let mu_over_c = (team_i.mu / c).exp();
                for (q, team_q) in team_ratings.iter().enumerate() {
                    let mu_c_over_sum_q = mu_over_c / sum_q[q];
                    if team_q.rank <= team_i.rank {
                        delta += (mu_c_over_sum_q * (1.0 - mu_c_over_sum_q)) / a[q];
                        if q == i {
                            omega += (1.0 - mu_c_over_sum_q) / a[q]
                        } else {
                            omega -= mu_c_over_sum_q / a[q]
                        }
                    }
                }
                omega *= team_i.sigma_sq / c;
                delta *= team_i.sigma_sq / c.powi(2);
                delta *= (self.gamma)(c, num_teams, team_i);

                update_team_rating(team_i, omega, delta, self.kappa)
            })
            .collect()
    }
}

fn calc_c(teams: &Vec<TeamRating>, beta_squared: f64) -> f64 {
    teams
        .iter()
        .map(|team| team.sigma_sq + beta_squared)
        .sum::<f64>()
        .sqrt()
}

fn calc_sum_q(teams: &Vec<TeamRating>, c: f64) -> Vec<f64> {
    let mut res = vec![0.0; teams.len()];
    for team_i in teams.iter() {
        let summed = (team_i.mu / c).exp();
        for (q, _) in teams
            .iter()
            .enumerate()
            .filter(|(_, team_q)| team_i.rank >= team_q.rank)
        {
            res[q] += summed;
        }
    }
    res
}

fn calc_a(teams: &Vec<TeamRating>) -> Vec<f64> {
    teams
        .iter()
        .map(|team| {
            teams
                .iter()
                .filter(|team_q| team.rank == team_q.rank)
                .count() as f64
        })
        .collect()
}

#[cfg(test)]
mod test {
    use crate::{
        env::test::env_model_test_func,
        env_builder::EnvBuilder,
        model::kind::ModelKind,
        rating::{GameResult, Rating},
    };

    #[test]
    fn case_solo_game_does_not_change_rating() {
        env_model_test_func(
            GameResult::new(vec![vec![Rating::default()]], vec![1]),
            Ok(vec![vec![Rating::default()]]),
            EnvBuilder::default().model(ModelKind::PlackettLuce).build(),
        );
    }

    #[test]
    fn case_2p_ffa() {
        env_model_test_func(
            GameResult::new(
                vec![vec![Rating::default()], vec![Rating::default()]],
                vec![1, 2],
            ),
            Ok(vec![
                vec![Rating::new(27.63523138347365, 8.065506316323548)],
                vec![Rating::new(22.36476861652635, 8.065506316323548)],
            ]),
            EnvBuilder::default().model(ModelKind::PlackettLuce).build(),
        );
    }

    #[test]
    fn case_3p_ffa() {
        env_model_test_func(
            GameResult::new(
                vec![
                    vec![Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default()],
                ],
                vec![1, 2, 3],
            ),
            Ok(vec![
                vec![Rating::new(27.868876552746237, 8.204837030780652)],
                vec![Rating::new(25.717219138186557, 8.057829747583874)],
                vec![Rating::new(21.413904309067206, 8.057829747583874)],
            ]),
            EnvBuilder::default().model(ModelKind::PlackettLuce).build(),
        );
    }

    #[test]
    fn case_4p_ffa() {
        env_model_test_func(
            GameResult::new(
                vec![
                    vec![Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default()],
                ],
                vec![1, 2, 3, 4],
            ),
            Ok(vec![
                vec![Rating::new(27.795084971874736, 8.263160757613477)],
                vec![Rating::new(26.552824984374855, 8.179213704945203)],
                vec![Rating::new(24.68943500312503, 8.083731307186588)],
                vec![Rating::new(20.96265504062538, 8.083731307186588)],
            ]),
            EnvBuilder::default().model(ModelKind::PlackettLuce).build(),
        );
    }

    #[test]
    fn case_5p_ffa() {
        env_model_test_func(
            GameResult::new(
                vec![
                    vec![Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default()],
                ],
                vec![1, 2, 3, 4, 5],
            ),
            Ok(vec![
                vec![Rating::new(27.666666666666668, 8.290556877154474)],
                vec![Rating::new(26.833333333333332, 8.240145629781066)],
                vec![Rating::new(25.72222222222222, 8.179996679645559)],
                vec![Rating::new(24.055555555555557, 8.111796013701358)],
                vec![Rating::new(20.72222222222222, 8.111796013701358)],
            ]),
            EnvBuilder::default().model(ModelKind::PlackettLuce).build(),
        );
    }

    #[test]
    fn case_3_teams_different_sized_players() {
        env_model_test_func(
            GameResult::new(
                vec![
                    vec![Rating::default(), Rating::default(), Rating::default()],
                    vec![Rating::default()],
                    vec![Rating::default(), Rating::default()],
                ],
                vec![1, 2, 3],
            ),
            Ok(vec![
                vec![
                    Rating::new(25.939870821784513, 8.247641552260456),
                    Rating::new(25.939870821784513, 8.247641552260456),
                    Rating::new(25.939870821784513, 8.247641552260456),
                ],
                vec![Rating::new(27.21366020491262, 8.274321317985242)],
                vec![
                    Rating::new(21.84646897330287, 8.213058173195341),
                    Rating::new(21.84646897330287, 8.213058173195341),
                ],
            ]),
            EnvBuilder::default().model(ModelKind::PlackettLuce).build(),
        );
    }
}
