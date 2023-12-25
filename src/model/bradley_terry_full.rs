use crate::{
    math::score,
    rating::{to_team_ratings, GammaFunc, Rating},
    utils::{update_team_rating, zip_without_self},
};

use super::model::Model;

pub struct BradleyTerryFull {
    gamma: GammaFunc,
    epsilon: f64,
    two_beta_sq: f64,
}

impl BradleyTerryFull {
    pub fn new(beta: f64, epsilon: f64, gamma: GammaFunc) -> Self {
        Self {
            gamma,
            epsilon,
            two_beta_sq: 2.0 * beta.powi(2),
        }
    }
}

impl Model for BradleyTerryFull {
    fn rate(&self, teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Vec<Vec<Rating>> {
        let team_ratings = to_team_ratings(&teams, Some(ranks));
        let num_teams = team_ratings.len() as f64;

        zip_without_self(&team_ratings)
            .map(|(team_i, others)| {
                let (omega, delta) = others.iter().fold((0., 0.), |(omega, delta), team_q| {
                    let ciq = (team_i.sigma_sq + team_q.sigma_sq + self.two_beta_sq).sqrt();
                    let piq = 1.0 / (1.0 + ((team_q.mu - team_i.mu) / ciq).exp());
                    let sigma_sq_to_ciq = team_i.sigma_sq / ciq;
                    let i_gamma = (self.gamma)(ciq, num_teams, team_i);
                    (
                        omega + sigma_sq_to_ciq * (score(team_q.rank, team_i.rank) - piq),
                        delta + ((i_gamma * sigma_sq_to_ciq) / ciq) * piq * (1.0 - piq),
                    )
                });

                update_team_rating(team_i, omega, delta, self.epsilon)
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
                .model(ModelKind::BradleyTerryFull)
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
                .model(ModelKind::BradleyTerryFull)
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
                vec![Rating::new(30.2704627669473, 7.788474807872566)],
                vec![Rating::new(25.0, 7.788474807872566)],
                vec![Rating::new(19.7295372330527, 7.788474807872566)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryFull)
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
                vec![Rating::new(32.90569415042095, 7.5012190693964005)],
                vec![Rating::new(27.63523138347365, 7.5012190693964005)],
                vec![Rating::new(22.36476861652635, 7.5012190693964005)],
                vec![Rating::new(17.09430584957905, 7.5012190693964005)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryFull)
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
                vec![Rating::new(35.5409255338946, 7.202515895247076)],
                vec![Rating::new(30.2704627669473, 7.202515895247076)],
                vec![Rating::new(25.0, 7.202515895247076)],
                vec![Rating::new(19.729537233052703, 7.202515895247076)],
                vec![Rating::new(14.4590744661054, 7.202515895247076)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryFull)
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
                    Rating::new(25.992743915179297, 8.19709997489984),
                    Rating::new(25.992743915179297, 8.19709997489984),
                    Rating::new(25.992743915179297, 8.19709997489984),
                ],
                vec![Rating::new(28.48909130001799, 8.220848339985736)],
                vec![
                    Rating::new(20.518164784802714, 8.127515465304823),
                    Rating::new(20.518164784802714, 8.127515465304823),
                ],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryFull)
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
                .model(ModelKind::BradleyTerryFull)
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
                vec![Rating::new(35.5409255338946, 7.993052538854532)],
                vec![Rating::new(30.2704627669473, 7.993052538854532)],
                vec![Rating::new(25.0, 7.993052538854532)],
                vec![Rating::new(19.729537233052703, 7.993052538854532)],
                vec![Rating::new(14.4590744661054, 7.993052538854532)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::BradleyTerryFull)
                .gamma(|_, k, _| 1.0 / k)
                .build(),
        );
    }
}
