Basketball Factors (other)

double digit spread
fatigue level (4 games in 6 nights)
offensive efficiency per 100 possessions = 100*(Points Scored)/(Possessions)
defensive efficiency per 100 possessions = 100*(Points Allowed/Possessions)



To Try:

1. MLP 10-2-2-1, or other
Image overlay and CNN analysis
    https://www.basketball-reference.com/boxscores/shot-chart/200810280BOS.html


2. What are the player stats for the starting lineup?

    Who are the players?
        get top level game record
    What are their stats?
        get top level game records of previous n games.
        
            get averages player fields
            get recent-weighted player fields?
            
3. Use number of days since start of season

    Get all games for all seasons
    Get first game after September of every year.
    Count number of days since the first day of season. Make column.             
                
4. Use Boosting/Bagging among all classifiers.

6. Use caching data with upfront cache invalidator.
Status: Done

7. NUM wins in last num_histories
Status: Done

10. Calc last 7 days travel distance.
Status: Done

11. Add OEFF, and DEFF to team history.
Status: Done

12. create average of center, forwards, and guards and make 3 additional columns remove all other
player history columns.
Status: Done

14. Do multi-year span.
Status: Done

15. Add player column: True or False for is_benched_this_game. Use zero minutes to infer injured.

16. Add losing streak
Status: Done

Add game_score = PTS + 0.4 * FG - 0.7 * FGA - 0.4*(FTA - FT) + 0.7 * ORB + 0.3 * DRB + STL + 0.7 * AST + 0.7 * BLK - 0.4 * PF - TOV
Status: Done

distance traveled in the last 7 days
Status: Done

Drop low probabs and rescore for all.
Status: Done

Create averages that weight more heavily to last 3 games.

RECORD FOR SEASON.

RECORD AGAINST same team in last N matchups.

5. Add:

    player's time in profession
    player's total career played minutes
    player's career points
    

Add total games played
Add total minutes played
Add total minutes played for team
Add days into season
seconds since last zero minutes game played

Develop algo to predict seconds played in game
    & calculate the points/second scored
    then multiply seconds by points/second

Did win last game
Points over or under last game.
Did win last game plus by (coeff * difference) in rank (win rate)
Spread movement (diff) between open and close.

record last 7 matchups

Average age of players
Status: Effectively set.

Average age of starting lineup

Find diff on mean games score per starting lineup position, per index position

Find diff of mean of all players in opposing team.
Status: Done

Number of returning players
mean Margin of victory
New/Returning players + quality of player

Starting players in last N history, out or injured. (use who is not going to play as injured?)
Percent players on injury list
percent of position players on injury

Score where agreement prediction for both teams in same match.

Fix (home & away)_starting_* fields -- all zero.

Appendix B: Player Meaning Technique

Options
1. Take mean of player for last n games.
    If player did not play in one of n games then:
        Use mean of player for last m games played. 
            If player has no history then:
                Use mean of average player.
                
2. For each feature create model that predicts the next game's metric
