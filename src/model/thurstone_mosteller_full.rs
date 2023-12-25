use crate::math::{v, vt, w, wt};
use crate::model::model::Model;
use crate::rating::{to_team_ratings, GammaFunc, Rating};
use crate::utils::{update_team_rating, zip_without_self};

pub struct ThurstoneMostellerFull {
    kappa: f64,
    gamma: GammaFunc,
    two_beta_sq: f64,
}

impl ThurstoneMostellerFull {
    pub fn new(beta: f64, kappa: f64, gamma: GammaFunc) -> Self {
        Self {
            kappa,
            gamma,
            two_beta_sq: 2.0 * beta.powi(2),
        }
    }
}

impl Model for ThurstoneMostellerFull {
    fn rate(&self, teams: Vec<Vec<Rating>>, ranks: Vec<usize>) -> Vec<Vec<Rating>> {
        let team_ratings = to_team_ratings(&teams, Some(ranks));
        zip_without_self(&team_ratings)
            .map(|(team_i, others)| {
                let (omega, delta) = others.iter().fold((0., 0.), |(omega, delta), team_q| {
                    let ciq = (team_i.sigma_sq + team_q.sigma_sq + self.two_beta_sq).sqrt();
                    let delta_mu = (team_i.mu - team_q.mu) / ciq;
                    let sigma_sq_to_ciq = team_i.sigma_sq / ciq;
                    let i_gamma = (self.gamma)(ciq, team_ratings.len() as f64, &team_i);
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
                .model(ModelKind::ThurstoneMostellerFull)
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
                vec![Rating::new(29.20524620886059, 7.632833464033909)],
                vec![Rating::new(20.79475379113941, 7.632833464033909)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
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
                vec![Rating::new(33.41049241772118, 6.861184222487201)],
                vec![Rating::new(25.0, 6.861184222487201)],
                vec![Rating::new(16.58950758227882, 6.861184222487201)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
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
                vec![Rating::new(37.61573862658177, 5.99095578185474)],
                vec![Rating::new(29.20524620886059, 5.99095578185474)],
                vec![Rating::new(20.79475379113941, 5.99095578185474)],
                vec![Rating::new(12.38426137341823, 5.99095578185474)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
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
                vec![Rating::new(41.82098483544236, 4.970639136506507)],
                vec![Rating::new(33.41049241772118, 4.970639136506507)],
                vec![Rating::new(25.0, 4.970639136506507)],
                vec![Rating::new(16.58950758227882, 4.970639136506507)],
                vec![Rating::new(8.17901516455764, 4.970639136506507)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
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
                    Rating::new(25.72407717517514, 8.154234192355084),
                    Rating::new(25.72407717517514, 8.154234192355084),
                    Rating::new(25.72407717517514, 8.154234192355084),
                ],
                vec![Rating::new(34.0010841338675, 7.7579369709569805)],
                vec![
                    Rating::new(15.274838690957358, 7.373381474061001),
                    Rating::new(15.274838690957358, 7.373381474061001),
                ],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
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
                vec![Rating::new(29.20524620886059, 7.784759515283723)],
                vec![Rating::new(20.79475379113941, 7.784759515283723)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
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
                vec![Rating::new(41.82098483544236, 7.436215601407348)],
                vec![Rating::new(33.41049241772118, 7.436215601407348)],
                vec![Rating::new(25.0, 7.436215601407348)],
                vec![Rating::new(16.58950758227882, 7.436215601407348)],
                vec![Rating::new(8.17901516455764, 7.436215601407348)],
            ]),
            EnvBuilder::default()
                .model(ModelKind::ThurstoneMostellerFull)
                .gamma(|_, k, _| 1. / k)
                .build(),
        );
    }
}
