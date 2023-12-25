use crate::{
    env_builder::EnvBuilder,
    error::{OpenSkillError, Result},
    model::model::Model,
    predict_draw::predict_draw,
    predict_win::predict_win,
    rating::{GameResult, OrdinalFunc, Rating},
};

pub struct Env {
    default_mu: f64,      // 25.0
    z: f64,               // 3.0
    default_sigma: f64,   // Mu/Z
    beta: f64,            // Sigma/2.0
    ordinal: OrdinalFunc, // Mu - Z * Sigma
    model: Box<dyn Model>,
}

impl Env {
    pub fn new(
        model: Box<dyn Model>,
        z: f64,
        default_mu: f64,
        default_sigma: f64,
        beta: f64,
        ordinal: OrdinalFunc,
    ) -> Self {
        Self {
            model,
            z,
            default_mu,
            default_sigma,
            beta,
            ordinal,
        }
    }

    pub fn default() -> Self {
        EnvBuilder::default().build()
    }

    pub fn new_rating(&self) -> Rating {
        Rating::new(self.default_mu, self.default_sigma)
    }

    pub fn rate(&self, result: &GameResult) -> Result<Vec<Vec<Rating>>> {
        let teams = result.teams.clone();
        let ranks = result.ranks.clone();

        if result.teams.is_empty() {
            return Err(OpenSkillError::InvalidTeamCount("0"));
        }

        if result.teams.len() < 2 {
            return Ok(result.teams.clone());
        }

        for team in &result.teams {
            if team.is_empty() {
                return Err(OpenSkillError::EmptyTeams);
            }
        }

        Ok(self.model.rate(teams, ranks))
    }

    pub fn rate_with_tau(&self, result: &GameResult, tau: f64) -> Result<Vec<Vec<Rating>>> {
        let mut teams = result.teams.clone();
        let ranks = result.ranks.clone();

        if result.teams.is_empty() {
            return Err(OpenSkillError::InvalidTeamCount("0"));
        }

        if result.teams.len() < 2 {
            return Ok(result.teams.clone());
        }

        for team in &result.teams {
            if team.is_empty() {
                return Err(OpenSkillError::EmptyTeams);
            }
        }

        let tau_squad = tau.powi(2);
        teams = teams
            .into_iter()
            .map(|team| {
                team.iter()
                    .map(|it| Rating::new(it.mu, (it.sigma.powi(2) + tau_squad).sqrt()))
                    .collect()
            })
            .collect();

        Ok(self.model.rate(teams, ranks))
    }

    pub fn ordinal(&self, rating: &Rating) -> f64 {
        (self.ordinal)(rating, self.z)
    }

    pub fn predict_draw(&self, team_ratings: &Vec<Vec<Rating>>) -> Result<f64> {
        predict_draw(team_ratings, self.beta)
    }

    pub fn predict_win(&self, team_ratings: &Vec<Vec<Rating>>) -> Result<Vec<f64>> {
        predict_win(team_ratings, self.beta)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use approx::relative_eq;

    use crate::{
        env::Env,
        error::Result,
        rating::{GameResult, Rating},
    };

    pub(crate) fn env_model_test_func(input: GameResult, want: Result<Vec<Vec<Rating>>>, env: Env) {
        let got = env.rate(&input);
        match (got.clone(), want.clone()) {
            (Ok(got), Ok(want)) => {
                assert_eq!(got.len(), want.len());
                got.iter().zip(want.iter()).for_each(|(g, w)| {
                    assert_eq!(g.len(), w.len());
                    g.iter().zip(w.iter()).for_each(|(g, w)| {
                        assert!(relative_eq!(g.mu, w.mu, epsilon = 1e-6), "mu: {g}, {w}");
                        assert!(
                            relative_eq!(g.sigma, w.sigma, epsilon = 1e-6),
                            "sigma: {g}, {w}"
                        );
                    })
                });
            }
            (Err(got), Err(want)) => {
                assert_eq!(got, want, "{got}, {want}");
            }
            _ => assert!(false, "{got:?}, {want:?}"),
        }
    }
}
