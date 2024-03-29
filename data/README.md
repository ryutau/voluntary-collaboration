# Data Descriptions
We have the following two files in this directory:

1. `main_exp_result.csv`: This file contains behavioral data from our main task (threshold public games) of the main experiment. Each row represents the result of a participant for a specific round. The columns in this file are detailed as follows:

   - **pid**: Represents the participant's identification code. Use this when merging with `participant_attributes.csv`.
   - **p_option**: Indicates whether group participation is mandatory ("A") or voluntary ("F") for the round.
   - **thr**: Shows the threshold value of the round.
   - **action**: Describes the action chosen by the participant. The possible actions are:
     - Cooperate ("C")
     - Defect ("D")
     - Leave ("L")
   - **belief_c**: Indicates the participant's estimation of the number of other players choosing the action 'cooperate'.
   - **belief_d**: Indicates the participant's estimation of the number of other players choosing to 'defect'.
   - **belief_l**: Indicates the participant's estimation of the number of other players deciding to 'leave'. This field will be NULL if group participation is mandatory.
   - **confidence**: Refers to the participant's self-reported confidence regarding their estimations.
   - **gamma_c**: This is calculated as the ratio of `belief_c` to 30.
   - **gamma_l**: This is the ratio of `belief_l` to 30.
   - **is_coop**: A boolean field that returns true if `action` is "C".
   - **is_leave**: A boolean field that returns true if `action` is "L".

2. `main_participant_attributes.csv`: This file contains the attributes of the participants in the main experiment, including economic, psychological, or demographic attributes elicited from the sub tasks and the postexperimental questionnaire. Each row represents attributes of a participant. The columns in this file are detailed as follows:

   - **pid**: Represents the participant's identification code. Use this when merging with `exp_result.csv`.
   - **fs_alpha**: Represents the participant's aversion to disadvantageous inequity, as derived from Fehr and Schmidt's utility function (FS-model).
   - **fs_beta**: Indicates the participant's aversion to advantageous inequity, as measured by the FS-model.
   - **r_from_hl**: Indicates the participant's risk attitude, as measured by the  methodology in Holt and Laury, 2002.
   - **iri_EC**: Indicates the participant's tendency to experience feelings of sympathy and compassion for unfortunate others assessed by the empathetic concern (EC) scale in Davis, 1980.
   - **iri_FS**: Indicates the participant's tendency to imaginatively transpose oneself into fictional situations measured by the fantasy (FS) scale in Davis, 1980.
   - **iri_PD**: Indicates the participant's tendency to experience distress and discomfort in response to extreme distress in others assessed by the personal distress (PD) scale in Davis, 1980.
   - **iri_PT**: Indicates the participant's tendency to spontaneously adopt the psychological point of view of others in everyday life measured by the perspective taking (PT) scale in Davis, 1980.
   - **siut**: Denotes the participant's intolerance against uncertainty measured by short intolerance of uncertainty scale in Carleton et al., 2007.
   - **crt**: Indicates the participant's tendency to suppress an intuitive but incorrect answer and come to a more deliberate and correct answer, as measured by the cognitive reflection tests adopted from Baron et al., 2015. and Toplak et al., 2014.
   - **general trust**: Indicates the participant's beliefs about honesty and trustworthiness of others, in general, assessed by questions from Yamagishi and Yamagishi, 1994.
   - **gender**: Refers the participant's gender (male, female, or other).
   - **age**: Refers the participant's age.
   - **t_order**: Denotes the order of conditions the participant was assigned to. Note that the ordering of the six conditions was (partially) randomized across participants: Each session was set to initiate with either the voluntary or the mandatory conditions, and within each of the voluntary and mandatory conditions, there were 6 possible permutations, stemming from combinations of the values (2, 4, 5). t_order represents each participant's sequence of conditions:
     - We employ a base-6 numbering system.
     - The first digit stands for the sequence of threshold values within the voluntary conditions. It ranges from 0 to 5, corresponding to the permutations: [(2, 4, 5), (2, 5, 4), (4, 2, 5), (4, 5, 2), (5, 2, 4), (5, 4, 2)].
     - The second digit is indicative of the sequence of threshold values for the mandatory conditions.
     - The third digit in this base-6 number is set to 1 if the session started with the mandatory conditions and 0 if it started with the voluntary conditions.
     - This base-6 number is then converted into its decimal equivalent, rangeing from 0 to 71.

3. `sub_exp_result.csv`: This file contains behavioral data from the additional experiment. Each row represents the result of a participant for a specific round. The columns in this file are detailed as follows:

   - **pid**: Represents the participant's identification code. Use this when merging with `sub_participant_attributes.csv`.
   - **rho**: Can be either 0, 0.5, or 1, and indicates the degree of loners' externality (i.e., the probability of a loner being included in the five-person groups) for the round.
   - **belief_c**: Indicates the participant's estimation of the number of other players choosing the action 'cooperate'.
   - **belief_d**: Indicates the participant's estimation of the number of other players choosing to 'defect'.
   - **belief_l**: Indicates the participant's estimation of the number of other players deciding to 'leave'. This field will be NULL if group participation is mandatory.
   - **action**: Describes the action chosen by the participant. The possible actions are:
     - Cooperate ("C")
     - Defect ("D")
     - Leave ("L")
   - **is_coop**: A boolean field that returns true if `action` is "C".
   - **round_no**: Integer indicating the participant's current round number.

4. `sub_participant_attributes.csv`: This file contains the attributes of the participants in the additional experiment. The columns in this file are detailed as follows:

   - **pid**: Represents the participant's identification code. Use this when merging with `sub_exp_result.csv`.
   - **age**: Indicates the participant's age.
   - **gender**: Indicatess the participant's gender (male: 0, female: 1, or not to declare: 3).
   - **session_no**: Indicates the session number that the participant involved.
   - **gamma**: Indicates the participant's belief about the effective cooperation computed from belief_c, belief_d, belief_l, and rho.

*) Data files were updated in March 2024 to correct column names and index numbers.
