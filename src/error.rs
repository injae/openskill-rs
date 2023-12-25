use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenSkillError {
    #[error("Empty teams")]
    EmptyTeams,
    #[error("Invalid team count {0}")]
    InvalidTeamCount(&'static str),
}

pub type Result<T> = std::result::Result<T, OpenSkillError>;
