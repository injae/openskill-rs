use statrs::distribution::{Continuous, ContinuousCDF, Normal};

pub(crate) fn normal() -> Normal {
    Normal::new(0.0, 1.0).unwrap()
}

pub(crate) fn draw_margin(num_teams: usize, total_player: usize, beta: f64) -> f64 {
    (total_player as f64).sqrt() * beta * normal().inverse_cdf((1.0 + 1.0 / num_teams as f64) / 2.0)
}

pub(crate) fn denominator(num_teams: usize, n: usize) -> f64 {
    (num_teams * (num_teams - 1)) as f64 / n as f64
}

pub(crate) fn sigma_bar(sigma_a: f64, sigma_b: f64, beta_squared: f64, total_player: usize) -> f64 {
    ((total_player as f64 * beta_squared) + sigma_a + sigma_b).sqrt()
}

pub(crate) fn score(rank1: usize, rank2: usize) -> f64 {
    if rank1 < rank2 {
        0.0
    } else if rank1 > rank2 {
        1.0
    } else {
        0.5
    }
}
pub(crate) fn v(x: f64, t: f64) -> f64 {
    let xt = x - t;
    let denom = normal().cdf(xt);
    if denom < f64::EPSILON {
        -xt
    } else {
        normal().pdf(xt) / denom
    }
}

pub(crate) fn w(x: f64, t: f64) -> f64 {
    let xt = x - t;
    let denom = normal().cdf(xt);
    if denom < f64::EPSILON {
        if x < 0. {
            1.
        } else {
            0.
        }
    } else {
        v(x, t) * (v(x, t) + xt)
    }
}

pub(crate) fn vt(x: f64, t: f64) -> f64 {
    let xx = x.abs();
    let b = normal().cdf(t - xx) - normal().cdf(-t - xx);
    if b < 1e-5 {
        if x < 0. {
            -x - t
        } else {
            -x + t
        }
    } else {
        let a = normal().pdf(t - xx) - normal().pdf(-t - xx);
        if x < 0. {
            -a / b
        } else {
            a / b
        }
    }
}

pub(crate) fn wt(x: f64, t: f64) -> f64 {
    let xx = x.abs();
    let b = normal().cdf(t - xx) - normal().cdf(-t - xx);
    if b < f64::EPSILON {
        1.0
    } else {
        ((t - xx) * normal().pdf(t - xx) + (t + xx) * normal().pdf(-t - xx)) / b
            + vt(x, t) * vt(x, t)
    }
}
