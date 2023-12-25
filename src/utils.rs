use crate::rating::{Rating, TeamRating};

pub fn zip_without_self<T>(vector: &Vec<T>) -> impl Iterator<Item = (&T, Vec<T>)>
where
    T: Clone,
{
    vector.iter().enumerate().map(|(i, it)| {
        (
            it,
            vector
                .iter()
                .enumerate()
                .filter_map(|(j, jt)| if i != j { Some(jt.clone()) } else { None })
                .collect::<Vec<T>>(),
        )
    })
}

pub(crate) fn update_team_rating(
    team: &TeamRating,
    omega: f64,
    delta: f64,
    epsilon: f64,
) -> Vec<Rating> {
    team.members
        .iter()
        .map(|it| {
            let sigma_sq = it.sigma.powi(2);
            Rating::new(
                it.mu + (sigma_sq / team.sigma_sq) * omega,
                it.sigma
                    * (1.0 - (sigma_sq / team.sigma_sq) * delta)
                        .max(epsilon)
                        .sqrt(),
            )
        })
        .collect()
}
