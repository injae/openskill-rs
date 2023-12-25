use crate::math::{v, vt, w, wt};
use crate::model::model::Model;
use crate::rating::{ladder_pairs, to_team_ratings, GammaFunc, Rating};
use crate::utils::update_team_rating;
use std::iter::zip;

pub struct ThurstoneMostellerPart {
    kappa: f64,
    gamma: GammaFunc,
    two_beta_sq: f64,
}

impl ThurstoneMostellerPart {
    pub fn new(beta: f64, kappa: f64, gamma: GammaFunc) -> Self {
        Self {
            kappa,
            gamma,
            two_beta_sq: 2.0 * beta.powi(2),
        }
    }
}

impl Model for ThurstoneMostellerPart {
    fn rate(&self, teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Vec<Vec<Rating>> {
        let team_ratings = to_team_ratings(&teams, Some(ranks));
        let num_teams = team_ratings.len() as f64;
        let adjacent_teams = ladder_pairs(&team_ratings);
        zip(&team_ratings, adjacent_teams)
            .map(|(team_i, adj_i)| {
                let (omega, delta) = adj_i.iter().fold((0., 0.), |(omega, delta), team_q| {
                    let ciq = 2. * (team_i.sigma_sq + team_q.sigma_sq + self.two_beta_sq).sqrt();
                    let delta_mu = (team_i.mu - team_q.mu) / ciq;
                    let sigma_sq_to_ciq = team_i.sigma_sq / ciq;
                    let i_gamma = (self.gamma)(ciq, num_teams, &team_i);
                    if team_q.rank == team_i.rank {
                        (
                            omega + sigma_sq_to_ciq * vt(delta_mu, self.kappa / ciq),
                            delta
                                + ((i_gamma * sigma_sq_to_ciq) / ciq)
                                    * wt(delta_mu, self.kappa / ciq),
                        )
                    } else {
                        let sign = if team_q.rank > team_i.rank { 1. } else { -1. };
                        (
                            omega + sign * sigma_sq_to_ciq * v(sign * delta_mu, self.kappa / ciq),
                            delta
                                + ((i_gamma * sigma_sq_to_ciq) / ciq)
                                    * w(sign * delta_mu, self.kappa / ciq),
                        )
                    }
                });
                update_team_rating(&team_i, omega, delta, self.kappa)
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
                .model(ModelKind::ThurstoneMostellerPart)
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
                vec![Rating::new(27.102616738180256, 8.24902473277454)],
                vec![Rating::new(22.897383261819744, 8.24902473277454)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
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
                vec![Rating::new(27.102616738180256, 8.24902473277454)],
                vec![Rating::new(25.0, 8.163845517855398)],
                vec![Rating::new(22.897383261819744, 8.24902473277454)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
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
                vec![Rating::new(27.102616738180256, 8.24902473277454)],
                vec![Rating::new(25.0, 8.163845517855398)],
                vec![Rating::new(25.0, 8.163845517855398)],
                vec![Rating::new(22.897383261819744, 8.24902473277454)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
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
                vec![Rating::new(27.102616738180256, 8.24902473277454)],
                vec![Rating::new(25.0, 8.163845517855398)],
                vec![Rating::new(25.0, 8.163845517855398)],
                vec![Rating::new(25.0, 8.163845517855398)],
                vec![Rating::new(22.897383261819744, 8.24902473277454)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
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
                    Rating::new(25.31287811922766, 8.309613085276991),
                    Rating::new(25.31287811922766, 8.309613085276991),
                    Rating::new(25.31287811922766, 8.309613085276991),
                ],
                vec![Rating::new(27.735657148831812, 8.257580565832717)],
                vec![
                    Rating::new(21.95146473194053, 8.245567434614435),
                    Rating::new(21.95146473194053, 8.245567434614435),
                ],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
                .build(),
        );
    }

    #[test]
    fn case_can_use_a_custom_gamma_with_k_2() {
        env_model_test_func(
            GameResult::new(
                vec![vec![Rating::default()], vec![Rating::default()]],
                vec![1, 2],
            ),
            Ok(vec![
                vec![Rating::new(27.102616738180256, 8.199631478529401)],
                vec![Rating::new(22.897383261819744, 8.199631478529401)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
                .gamma(|_, k, _| 1. / k)
                .build(),
        );
    }

    #[test]
    fn case_can_use_a_custom_gamma_with_k_5() {
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
                vec![Rating::new(27.102616738180256, 8.280111667130026)],
                vec![Rating::new(25.0, 8.226545690375827)],
                vec![Rating::new(25.0, 8.226545690375827)],
                vec![Rating::new(25.0, 8.226545690375827)],
                vec![Rating::new(22.897383261819744, 8.280111667130026)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerPart)
                .gamma(|_, k, _| 1. / k)
                .build(),
        );
    }
}
