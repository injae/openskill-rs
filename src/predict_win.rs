use crate::{
    error::OpenSkillError,
    math::{self, normal},
    rating::{to_team_ratings, Rating},
    utils::zip_without_self,
    validate::validate_team,
};

use statrs::distribution::ContinuousCDF;

pub fn predict_win(teams: &Vec<Vec<Rating>>, beta: f64) -> Result<Vec<f64>, OpenSkillError> {
    let beta_squared = beta.powi(2);

    validate_team(&teams)?;

    let num_teams = teams.len();
    let team_ratings = to_team_ratings(teams, None);
    let denom = math::denominator(num_teams, 2);

    Ok(zip_without_self(&team_ratings)
        .map(|(team_i, others)| {
            others
                .iter()
                .map(|team_q| {
                    normal().cdf(
                        ((*team_i).mu - team_q.mu)
                            / math::sigma_bar(
                                team_i.sigma_sq,
                                team_q.sigma_sq,
                                beta_squared,
                                team_i.team_size() + team_q.team_size(),
                            ),
                    )
                })
                .sum::<f64>()
                / denom
        })
        .collect())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::constant::DEFAULT_BETA;
    use approx::relative_eq;
    use std::iter::zip;

    fn predict_win_test(teams: Vec<Vec<Rating>>, want: Result<Vec<f64>, OpenSkillError>) {
        let got = predict_win(&teams, DEFAULT_BETA);
        match (got.clone(), want.clone()) {
            (Ok(got), Ok(want)) => {
                assert_eq!(got.len(), want.len());
                zip(got.iter(), want.iter()).for_each(|(g, w)| {
                    assert!(
                        relative_eq!(g, w, epsilon = 1e-6, max_relative = 1e-6),
                        "got: {g}, want: {w}",
                    )
                });
            }
            (Err(got), Err(want)) => {
                assert_eq!(got, want);
            }
            _ => assert!(false, "got: {got:?}, want: {want:?}"),
        };
    }

    #[test]
    fn predict_win_outcome_for_two_teams() {
        let teams = vec![
            vec![Rating::default(), Rating::new(32.444, 5.123)],
            vec![Rating::new(73.381, 1.421), Rating::new(25.188, 6.211)],
        ];
        let want = Ok(vec![0.0020706344961249385, 0.997929365503875]);
        predict_win_test(teams, want);
    }

    #[test]
    fn predict_win_outcome_for_multiple_asymmetric_teams() {
        let teams = vec![
            vec![Rating::default(), Rating::new(32.444, 5.123)],
            vec![Rating::new(73.381, 1.421), Rating::new(25.188, 6.211)],
            vec![Rating::new(32.444, 5.123)],
            vec![Rating::new(25.188, 6.211)],
        ];
        let want = Ok(vec![
            0.3273280055056081,
            0.49965489412719827,
            0.132583880271438,
            0.04043322009575564,
        ]);
        predict_win_test(teams, want);
    }

    #[test]
    fn predict_win_3_player_newbie_ffa() {
        let teams = vec![
            vec![Rating::default()],
            vec![Rating::default()],
            vec![Rating::default()],
        ];
        let want = Ok(vec![
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
        ]);
        predict_win_test(teams, want);
    }

    #[test]
    fn predict_win_4_player_newbie_ffa() {
        let teams = vec![
            vec![Rating::default()],
            vec![Rating::default()],
            vec![Rating::default()],
            vec![Rating::default()],
        ];
        let want = Ok(vec![0.25, 0.25, 0.25, 0.25]);
        predict_win_test(teams, want);
    }

    #[test]
    fn predict_win_4_players_of_varying_skill() {
        let teams = vec![
            vec![Rating::new(1.0, 0.1)],
            vec![Rating::new(2.0, 0.1)],
            vec![Rating::new(3.0, 0.1)],
            vec![Rating::new(4.0, 0.1)],
        ];
        let want = Ok(vec![
            0.18420221980528928,
            0.22786446154550788,
            0.27213553845449207,
            0.31579778019471066,
        ]);
        predict_win_test(teams, want);
    }
}
