#[cfg(test)]
mod test {
    use approx::relative_eq;

    use crate::{model::model::Model, rating::Rating};

    fn model_test_func(
        teams: Vec<Vec<Rating>>,
        ranks: Vec<usize>,
        want: Vec<Vec<Rating>>,
        model: Box<dyn Model>,
    ) {
        let got = model.rate(teams, ranks);
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
}
