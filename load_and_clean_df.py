import pandas as pd
import numpy as np

import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_and_clean_df():
    file_path = "IPL.csv"
    exclude_cols = ["Unnamed: 0", "match_type", "event_name", "gender", "team_type", "match_number", "power_surge_start", "balls_per_over", "overs"]

    # Load the latest version
    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "chaitu20/ipl-dataset2008-2025",
    file_path,
    pandas_kwargs={"usecols":lambda col: col not in exclude_cols,}
    )


    # date
    df['date'] = pd.to_datetime(df['date'])


    # batting_team
    df['batting_team'] = df['batting_team'].str.replace('Rising Pune Supergiants','Rising Pune Supergiant')
    df['batting_team'] = df['batting_team'].str.replace('Royal Challengers Bangalore',"Royal Challengers Bengaluru")

    df['batting_team'] = df['batting_team'].astype('string')


    # bowling_team
    df['bowling_team'] = df['bowling_team'].str.replace('Rising Pune Supergiants','Rising Pune Supergiant')
    df['bowling_team'] = df['bowling_team'].str.replace('Royal Challengers Bangalore',"Royal Challengers Bengaluru")

    df['bowling_team'] = df['bowling_team'].astype('string')


    # batter
    df['batter'] = df['batter'].astype('string')


    # balls_faced
    df['balls_faced'] = df['balls_faced'].astype(bool)


    # bowler
    df['bowler'] = df['bowler'].astype('string')


    # valid_ball
    df['valid_ball'] = df['valid_ball'].astype(bool)


    # extra_type
    df['extra_type'] = df['extra_type'].fillna('N/A').astype('string')


    # non_striker
    df['non_striker'] = df['non_striker'].astype('string')


    # wicket_kind
    df['wicket_kind'] = df['wicket_kind'].fillna('N/A').astype('string')


    # player_out
    df['player_out'] = df['player_out'].fillna('N/A').astype('string')


    # fielders
    df['fielders'] = df['fielders'].fillna('N/A').astype('string')


    # runs_target



    # review_batter
    df['review_batter'] = df['review_batter'].fillna('N/A').astype('string')


    # team_reviewed
    df['team_reviewed'] = df['team_reviewed'].str.replace('Royal Challengers Bangalore',"Royal Challengers Bengaluru")
    df['team_reviewed'] = df['team_reviewed'].fillna('N/A').astype('string')


    # review_decision
    df['review_decision'] = df['review_decision'].fillna('N/A').astype('string')


    # umpire
    df['umpire'] = df['umpire'].fillna('N/A').astype('string')


    # player_of_match
    df['player_of_match'] = df['player_of_match'].astype('string')


    # match_won_by
    df['match_won_by'] = df['match_won_by'].str.replace('Rising Pune Supergiants','Rising Pune Supergiant')
    df['match_won_by'] = df['match_won_by'].str.replace('Royal Challengers Bangalore',"Royal Challengers Bengaluru")

    df['is_abandoned'] = (df['match_won_by'] == "Unknown") & (df['superover_winner'].isna())

    df['match_won_by'] = df['match_won_by'].replace('Unknown', np.nan).fillna(df['superover_winner']).fillna('N/A').astype('string')


    # win_outcome
    def func(row):
        if row['is_abandoned']:
            return 'N/A'
        elif pd.notna(row['superover_winner']):
            return 'Superover'
        else:
            return row['win_outcome']

    df['win_outcome'] = df.apply( func, axis=1 )
    df['win_outcome'] = df['win_outcome'].astype('string')

    # Fast Approach
    # df['win_outcome'] = np.select(
    #     [df['is_abandoned'] == True, df['superover_winner'].notna()],
    #     ['N/A', 'Superover'],
    #     default=df['win_outcome']
    # )


    # toss_winner
    df['toss_winner'] = df['toss_winner'].str.replace('Rising Pune Supergiants','Rising Pune Supergiant')
    df['toss_winner'] = df['toss_winner'].str.replace('Royal Challengers Bangalore',"Royal Challengers Bengaluru")

    df['toss_winner'] = df['toss_winner'].astype('string')


    # toss_descision
    df['toss_decision'] = df['toss_decision'].astype('string')


    # venue
    df['venue'] = df['venue'].replace({
        # M Chinnaswamy Stadium
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',

        # Punjab Cricket Association Stadium
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh':
            'Punjab Cricket Association IS Bindra Stadium, Mohali',

        # Delhi Stadium
        'Feroz Shah Kotla': 'Feroz Shah Kotla Stadium',
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',

        # Hyderabad Stadium
        'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad':
            'Rajiv Gandhi International Stadium, Uppal',

        # Chennai Stadium
        'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk',
        'MA Chidambaram Stadium, Chepauk, Chennai':
            'MA Chidambaram Stadium, Chepauk',

        # Mumbai Stadium
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium',

        # Eden Gardens
        'Eden Gardens, Kolkata': 'Eden Gardens',

        # ACA-VDCA Stadium
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam':
            'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',

        # Jaipur Stadium
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',

        # Ahmedabad Stadium (renamed)
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium, Ahmedabad',

        # Abu Dhabi Stadium
        'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium',

        # Mullanpur Stadium
        'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh':
            'Maharaja Yadavindra Singh International Cricket Stadium',
        'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur':
            'Maharaja Yadavindra Singh International Cricket Stadium'
    })
    df['venue'] = df['venue'].astype('string')


    # city
    df['city'] = df['city'].replace({
        # Official name change
        'Bangalore': 'Bengaluru',
    })
    df['city'] = df['city'].astype('string')


    # season
    pd.set_option('future.no_silent_downcasting', True)

    df['season'] = df['season'].replace('2007/08', 1).replace('2009',2).replace(2009,2).replace('2009/10',3).replace('2011',4).replace(2011,4).replace(2012,5).replace(2013,6).replace(2014,7).replace(2015,8).replace(2016,9).replace(2017,10).replace(2018,11).replace(2019,12).replace('2019',12).replace('2020/21',13).replace('2021',14).replace(2021,14).replace(2022,15).replace(2023,16).replace(2024,17).replace(2025,18).replace(2026,19)

    df['season'] = df['season'].astype('int64')


    # superover_winner
    df['superover_winner'] = df['superover_winner'].str.replace('Royal Challengers Bangalore',"Royal Challengers Bengaluru")
    df['superover_winner'] = df['superover_winner'].fillna('N/A').astype('string')


    # result_type
    df['result_type'] = df['result_type'].fillna('N/A').astype('string')


    # method
    df['method'] = df['method'].replace('D/L', True).fillna(False).astype('bool')


    # stage
    df['stage'] = df['stage'].replace('Unknown', 'League match').replace("Elimination Final","Eliminator").astype('string')


    # event_match_no
    df['event_match_no'] = df['event_match_no'].replace('Unknown', 0).astype('int64')

    def f(row):
        if row['event_match_no'] == 0:
            return row['stage']
        else:
            return row['event_match_no']

    df['event_match_no'] = df.apply( f, axis=1 )


    # new_batter
    df['new_batter'] = df['new_batter'].fillna('N/A').astype('string')


    # next_batter
    df['next_batter'] = df['next_batter'].fillna('N/A').astype('string')

    # Renaming Section

    rename_dict = {
        "ball_no": "overs_balls",
        "bat_pos": "batter_position_this_match",
        "runs_batter":"runs_counted_for_batter",
        "balls_faced":"is_ball_counted_towards_batter",
        "valid_ball":"is_legal_ball",
        "runs_extras":"extra_runs_this_ball",
        "runs_total":"total_runs_this_ball",
        "runs_bowler":"runs_counted_for_bowler",
        "runs_not_boundary":"runs_greater_4_but_not_boundary",
        "non_striker_pos":"non_striker_position",
        "wicket_kind":"wicket_type",
        "fielders":"fielders_involved_in_wicket",
        "runs_target":"runs_target_this_match",
        "review_batter":"batter_involved_in_review",
        "umpire":"umpire_involved_in_review",
        "umpires_call":"is_umpires_call",
        "match_won_by":"winning_team",
        "venue":"stadium",
        "season":"ipl_season_no",
        "method":"is_DLS_applied",
        "event_match_no":"match_no_this_season",
        "team_runs":"team_runs_cumulative",
        "team_balls":"team_balls_cumulative",
        "team_wickets":"team_wickets_cumulative",
        "batter_runs":"batter_runs_cumulative",
        "batter_balls":"batter_balls_cumulative",
        "striker_out":"is_striker_out"
    }

    df.rename(columns=rename_dict, inplace=True)

    df['match_phase'] = pd.cut(df['over'], bins=[-1, 5, 15, 20],
                            labels=['Powerplay', 'Middle', 'Death'])
    


    df.attrs['col_description'] = {
        "match_id": "Unique identifier for each IPL match",
        "date": "Match date in datetime format",
        "innings": "Inning number counts from 1 to 2 (or more in case of super over)",
        "batting_team": "Team currently batting",
        "bowling_team": "Team currently bowling",
        "over": "Over number (0 to 19)",
        "ball": "Ball number within the over",
        "overs_balls": "Over number and ball number combined in broadcast format",
        "batter": "Name of the batsman on strike",
        "batter_position_this_match": "Batting position of present batter in strike (1 or 2 = opener, 3 to 7 = middle order, > 7 = tailender)",
        "runs_counted_for_batter": "Runs scored by strike batter on this delivery",
        "is_ball_counted_towards_batter": "Does this delivery count towards batter's total ball count",
        "bowler": "Name of the bowler bowling this delivery",
        "is_legal_ball": "1 if legal delivery, 0 if not legal delivery (e.g. wide, no-ball, etc)",
        "extra_runs_this_ball": "Runs scored as extras (byes, wides, no-balls) in this delivery",
        "total_runs_this_ball": "Total runs from this ball (batter runs + extras)",
        "runs_counted_for_bowler": "Runs counted against the bowler (excludes some type of extras like legbye extras, etc)",
        "runs_greater_4_but_not_boundary": "True when total runs in delivery are greater than 4 but not a boundary (e.g. overthrows, etc)",
        "extra_type": "Type of extra (wide, no-ball, bye, etc.)",
        "non_striker": "Name of the non-striker batter in this delivery",
        "non_striker_position": "Batting position of the non-striker in this match",
        "wicket_type": "Type of dismissal (bowled, caught, run out, etc.)",
        "player_out": "Name of player who got out (if any)",
        "fielders_involved_in_wicket": "Name of the fielders involved in the dismissal",
        "runs_target_this_match": "Target for the chasing team (only filled in the even innings number)",
        "batter_involved_in_review": "Batter who requested the review (if any)",
        "team_reviewed": "Team that took the DRS review (if any)",
        "review_decision": "Final decision after review (out/not out)",
        "umpire_involved_in_review": "Umpire who handled the review (if any)",
        "is_umpires_call": "Whether umpire's original decision stood (if any)",
        "player_of_match": "Name of the player who won the player of the match award for this match",
        "winning_team": "Team that won the match, Else N/A if no result",
        "win_outcome": "Victory detail (by runs, wickets, or other)",
        "toss_winner": "Team that won the toss this match",
        "toss_decision": "Decision took after winning toss (bat first or field first)",
        "stadium": "Stadium where the match was played",
        "city": "City where the match was played",
        "day": "Day of the month the match was played (1-31)",
        "month": "Month of the year the match was played (1-12)",
        "year": "The year in which the match was played (2008-)",
        "ipl_season_no": "Season number of the IPL (1-)",
        "superover_winner": "Name of the team which won the super over if match was decided via super over Else N/A",
        "result_type": "Match result type: N/A if normal, tie if scores equal before superover, no result if the match has been cancelled because of rain interuption",
        "is_DLS_applied": "True if DLS method was applied finally in this match to revise the chasing target in case of rain, else False",
        "match_no_this_season": "Match number of the match in the season (e.g., 43, 'Final', etc)",
        "stage": "Stage of the match in the season (League Match, Eliminator, Qualifier 1, Final, etc.)",
        "team_runs_cumulative": "Total cumulative runs scored by the team after this point/delivery",
        "team_balls_cumulative": "Total cumulative balls played by the team after this delivery",
        "team_wicket": "Cumulative wickets fallen of the batting team",
        "new_batter": "Name of the new batter entered the field after wicket and before this delivery (if any) else N/A",
        "batter_runs_cumulative": "Cumulative runs made by the striker batter after this delivery",
        "batter_balls_cumulative": "Cumulative balls faced by the striker batter after this delivery",
        "bowler_wicket": "1 if the wicket counts towards bowler (e.g. caught, bowled, etc) else 0 (e.g. run out, etc)",
        "batting_partners": "Current batting pair i.e. pair of strike batter and non strike batter",
        "next_batter": "Name of the next batter scheduled to come in (if dismissal/wicket ball) Else N/A",
        "is_striker_out": "Boolean: True if the player out this delivery is striker. False if the player out is non striker or no player is out.",
        "is_abandoned": "Boolean: True if the match is abandoned else False",
        "match_phase": "Phase of this delivery in the match : Powerplay, Middle, Death"
    }
    
    return df



if __name__ == "__main__":
    load_and_clean_df()