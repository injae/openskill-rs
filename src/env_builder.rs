use crate::{
    constant::*,
    env::Env,
    model::{
        bradley_terry_full::BradleyTerryFull, bradley_terry_part::BradleyTerryPart,
        kind::ModelKind, model::Model, plackett_luce::PlackettLuce,
        thurstone_mosteller_full::ThurstoneMostellerFull,
        thurstone_mosteller_part::ThurstoneMostellerPart,
    },
    rating::{default_gamma, default_ordinal, GammaFunc, OrdinalFunc},
};

#[derive(Default)]
pub struct EnvBuilder {
    beta: Option<f64>,
    gamma: Option<GammaFunc>,
    kappa: Option<f64>,
    model: Option<ModelKind>,
    mu: Option<f64>,
    ordinal: Option<OrdinalFunc>,
    sigma: Option<f64>,
    z: Option<f64>,
}

impl EnvBuilder {
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }

    pub fn gamma(mut self, gamma: GammaFunc) -> Self {
        self.gamma = Some(gamma);
        self
    }

    pub fn kappa(mut self, kappa: f64) -> Self {
        self.kappa = Some(kappa);
        self
    }

    pub fn model(mut self, model: ModelKind) -> Self {
        self.model = Some(model);
        self
    }

    pub fn mu(mut self, mu: f64) -> Self {
        self.mu = Some(mu);
        self
    }

    pub fn ordinal(mut self, ordinal: OrdinalFunc) -> Self {
        self.ordinal = Some(ordinal);
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    pub fn z(mut self, z: f64) -> Self {
        self.z = Some(z);
        self
    }

    pub fn build(self) -> Env {
        let mu = self.mu.unwrap_or(DEFAULT_MU);
        let z = self.z.unwrap_or(DEFAULT_Z);
        let sigma = self.sigma.unwrap_or(mu / z);
        let beta = self.beta.unwrap_or(sigma / 2.0);
        let kappa = self.kappa.unwrap_or(KAPPA);
        let gamma = self.gamma.unwrap_or(default_gamma);
        let ordinal = self.ordinal.unwrap_or(default_ordinal);
        let model: Box<dyn Model> = match self.model.unwrap_or_default() {
            ModelKind::PlackettLuce => Box::new(PlackettLuce::new(beta, kappa, gamma)),
            ModelKind::BradleyTerryFull => Box::new(BradleyTerryFull::new(beta, kappa, gamma)),
            ModelKind::BradleyTerryPart => Box::new(BradleyTerryPart::new(beta, kappa, gamma)),
            ModelKind::ThurstoneMostellerPart => {
                Box::new(ThurstoneMostellerPart::new(beta, kappa, gamma))
            }
            ModelKind::ThurstoneMostellerFull => {
                Box::new(ThurstoneMostellerFull::new(beta, kappa, gamma))
            }
        };

        Env::new(model, z, mu, sigma, beta, ordinal)
    }
}
