use crate::{
    math::score,
    rating::{ladder_pairs, to_team_ratings, GammaFunc, Rating},
    utils::update_team_rating,
};

use super::model::Model;
use std::iter::zip;

pub struct BradleyTerryPart {
    two_beta_sq: f64,
    kappa: f64,
    gamma: GammaFunc,
}

impl BradleyTerryPart {
    pub fn new(beta: f64, kappa: f64, gamma: GammaFunc) -> Self {
        Self {
            two_beta_sq: 2.0 * beta.powi(2),
            kappa,
            gamma,
        }
    }
}

impl Model for BradleyTerryPart {
    fn rate(&self, teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Vec<Vec<Rating>> {
        let team_ratings = to_team_ratings(&teams, Some(ranks));
        let adjacent_teams = ladder_pairs(&team_ratings);

        zip(&team_ratings, adjacent_teams)
            .map(|(team_i, adj)| {
                let (omega, delta) = adj.iter().fold((0., 0.), |(omega, delta), team_q| {
                    let ciq = (team_i.sigma_sq + team_q.sigma_sq + self.two_beta_sq).sqrt();
                    let piq = 1.0 / (1.0 + ((team_q.mu - team_i.mu) / ciq).exp());
                    let sigma_sq_to_ciq = team_i.sigma_sq / ciq;
                    let i_gamma = (self.gamma)(ciq, team_ratings.len() as f64, &team_i);
                    (
                        omega + sigma_sq_to_ciq * (score(team_q.rank, team_i.rank) - piq),
                        delta + ((i_gamma * sigma_sq_to_ciq) / ciq) * piq * (1.0 - piq),
                    )
                });

                update_team_rating(team_i, omega, delta, self.kappa)
            })
            .collect()
    }
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
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .build(),
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
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .build(),
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
                vec![Rating::new(27.63523138347365, 8.065506316323548)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(22.36476861652635, 8.065506316323548)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .build(),
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
                vec![Rating::new(27.63523138347365, 8.065506316323548)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(22.36476861652635, 8.065506316323548)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .build(),
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
                vec![Rating::new(27.63523138347365, 8.065506316323548)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(22.36476861652635, 8.065506316323548)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .build(),
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
                    Rating::new(25.219231461891965, 8.293401112661954),
                    Rating::new(25.219231461891965, 8.293401112661954),
                    Rating::new(25.219231461891965, 8.293401112661954),
                ],
                vec![Rating::new(28.48909130001799, 8.220848339985736)],
                vec![
                    Rating::new(21.291677238090045, 8.206896387427937),
                    Rating::new(21.291677238090045, 8.206896387427937),
                ],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .build(),
        );
    }

    #[test]
    fn case_use_a_custom_gamma_with_k_2() {
        env_model_test_func(
            GameResult::new(
                vec![vec![Rating::default()], vec![Rating::default()]],
                vec![1, 2],
            ),
            Ok(vec![
                vec![Rating::new(27.63523138347365, 8.122328620674137)],
                vec![Rating::new(22.36476861652635, 8.122328620674137)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .gamma(|_, k, _| 1.0 / k)
                .build(),
        );
    }

    #[test]
    fn case_use_a_custom_gamma_with_k_5() {
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
                vec![Rating::new(27.63523138347365, 8.249579113843055)],
                vec![Rating::new(25.0, 8.16496580927726)],
                vec![Rating::new(25.0, 8.16496580927726)],
                vec![Rating::new(25.0, 8.16496580927726)],
                vec![Rating::new(22.36476861652635, 8.249579113843055)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryPart)
                .gamma(|_, k, _| 1.0 / k)
                .build(),
        );
    }
}
