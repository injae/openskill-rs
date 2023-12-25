use crate::math::{self, normal};
use crate::rating::calc_total_player;
use crate::{
    error::OpenSkillError,
    rating::{to_team_ratings, Rating},
    validate::validate_team,
};
use itertools::Itertools;
use statrs::distribution::ContinuousCDF;

const ERR_MUST_CONTAIN_AT_LEAST_1_PLAYER: OpenSkillError =
    OpenSkillError::InvalidTeamCount("team must contain atleast 1 player");

pub fn predict_draw(teams: &Vec<Vec<Rating>>, beta: f64) -> Result<f64, OpenSkillError> {
    let beta_squared = beta.powi(2);
    validate_team(&teams)?;

    let num_teams = teams.len();
    match num_teams {
        0 => return Err(ERR_MUST_CONTAIN_AT_LEAST_1_PLAYER),
        1 => return Ok(1.0),
        _ => {}
    }

    let team_ratings = to_team_ratings(teams, None);
    let denom = math::denominator(num_teams, if num_teams > 2 { 1 } else { 2 });
    let total_player = calc_total_player(&team_ratings);
    let draw_margin = math::draw_margin(num_teams, total_player, beta);

    Ok(team_ratings
        .into_iter()
        .permutations(2)
        .map(|it| {
            let (mu_a, sigma_a) = (it[0].mu, it[0].sigma_sq);
            let (mu_b, sigma_b) = (it[1].mu, it[1].sigma_sq);
            let total_player = calc_total_player(&it);
            let sigma_bar = math::sigma_bar(sigma_a, sigma_b, beta_squared, total_player);
            normal().cdf((draw_margin - mu_a + mu_b) / sigma_bar)
                - normal().cdf((mu_a - mu_b - draw_margin) / sigma_bar)
        })
        .sum::<f64>()
        .abs()
        / denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constant::DEFAULT_BETA, rating::Rating};
    use approx::relative_eq;

    #[test]
    fn if_a_tree_falls_in_the_forest() {
        let teams = vec![];
        let want = Err(ERR_MUST_CONTAIN_AT_LEAST_1_PLAYER);
        predict_draw_test(teams, want);
    }

    #[test]
    fn predicts_100_percent_draw_for_solitaire() {
        let teams = vec![vec![Rating::new(32.444, 1.123)]];
        let want = Ok(1.0);
        predict_draw_test(teams, want);
    }

    #[test]
    fn predicts_100_percent_draw_for_self_v_self() {
        let teams = vec![
            vec![Rating::new(35.881, 0.0001)],
            vec![Rating::new(35.881, 0.0001)],
        ];
        let want = Ok(1.0);
        predict_draw_test(teams, want);
    }

    #[test]
    fn predicts_draw_for_two_teams() {
        let teams = vec![
            vec![Rating::default(), Rating::new(32.444, 1.123)],
            vec![Rating::new(35.881, 0.0001), Rating::new(25.188, 1.421)],
        ];
        let want = Ok(0.6948395180810698);
        predict_draw_test(teams, want);
    }

    #[test]
    fn predicts_draw_for_three_asymmetric_teams() {
        let teams = vec![
            vec![Rating::default(), Rating::new(32.444, 1.123)],
            vec![Rating::new(35.881, 0.0001), Rating::new(25.188, 1.421)],
            vec![Rating::default()],
            vec![Rating::new(32.444, 1.123)],
            vec![Rating::new(35.881, 0.0001)],
        ];
        let want = Ok(0.0835656139343231);
        predict_draw_test(teams, want);
    }

    fn predict_draw_test(teams: Vec<Vec<Rating>>, want: Result<f64, OpenSkillError>) {
        let beta = DEFAULT_BETA;
        let got = predict_draw(&teams, beta);
        match (got, want) {
            (Ok(got), Ok(want)) => {
                assert!(
                    relative_eq!(got, want, epsilon = 1e-6),
                    "got: {got}, want: {want}",
                );
            }
            (Err(got), Err(want)) => {
                assert_eq!(got, want, "got: {got}, want: {want}");
            }
            _ => assert!(false, "got: {got:?}, want: {want:?}"),
        };
    }
}
