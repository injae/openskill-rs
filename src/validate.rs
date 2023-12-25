use crate::{error::OpenSkillError, rating::Rating};

pub(crate) fn validate_team(teams: &Vec<Vec<Rating>>) -> Result<(), OpenSkillError> {
    if teams.iter().filter(|team| team.len() < 1).count() >= 1 {
        return Err(OpenSkillError::InvalidTeamCount(
            "team must contain atleast 1 player",
        ));
    }
    Ok(())
}
