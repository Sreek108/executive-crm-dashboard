# modules/etl_sql.py
import pandas as pd
import sqlalchemy as sa

def load_crm_tables(conn_str: str) -> dict:
    eng = sa.create_engine(conn_str)

    leads = pd.read_sql("""
        SELECT L.LeadId, L.FullName, L.Company, L.CountryId, L.CityRegionId,
               L.LeadStageId, L.LeadStatusId, L.LeadScoringId,
               L.CreatedOn, L.ModifiedOn
        FROM Lead L
        WHERE L.IsActive = 1
    """, eng)

    lead_status = pd.read_sql("SELECT LeadStatusId, StatusName_E FROM LeadStatus", eng)
    leads = leads.merge(lead_status, on='LeadStatusId', how='left')
    leads['Converted'] = (leads['StatusName_E'].fillna('').str.upper() == 'WON').astype(int)

    calls = pd.read_sql("""
        SELECT LC.LeadCallId, LC.LeadId, LC.CallDateTime, LC.DurationSeconds,
               LC.CallStatusId, LC.SentimentId
        FROM LeadCall LC
    """, eng)
    calls['IsSuccessful'] = (calls['CallStatusId'] == 1).astype(int)

    tasks = pd.read_sql("""
        SELECT S.ScheduleId, S.LeadId, S.TaskTypeId, S.ScheduledDate, S.TaskStatusId,
               S.AssignedAgentId, S.CompletedDate
        FROM Schedule S
    """, eng)
    tasks['IsBreach'] = (tasks['TaskStatusId'] == 5).astype(int)  # Overdue

    return {'leads': leads, 'calls': calls, 'tasks': tasks}
