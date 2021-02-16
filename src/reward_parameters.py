# hotspot detection
discount_factor = 0.25
trip_threshold = 3
hotspot_max_reward = 100

# rewards
unit_reward = 10
base_reward_threshold = 60
base_reward = 4000

target_increment = 5
reward_increment = 200
reward_boost_factor = 1.2

# costs
per_km_cost = 7.5
# opportunity cost represents the cost to the driver of being online on the platform as opposed to pursuing another
# means of income
opportunity_cost = 0.0

# reward weights - different driver clusters have different weights which indicate the value they attribute to different
# types of rewards
weights_length = 5  # number of weight options available
trip_reward_weights = [1, 1, 1, 1, 1]  # weight for trip reward
weekly_reward_weights = [1, 1.2, 0.8, 1.4, 0.6]  # weight for weekly reward
opportunity_cost_weights = [1, 0.8, 1.2, 0.6, 1.4]  # weight for opportunity_cost

# hourly weights - drivers are more/less likely to accespt a trip based on time of day
# this additional reward attempts to make drivers mimic the accceptance pattern in training data
base_hourly_reward_weight = 500
hourly_acceptance_rates = [0.71085678, 0.71566464, 0.6707093 , 0.63993246, 0.59757785,
       0.61685374, 0.50163666, 0.4979832 , 0.56854489, 0.7444813 ,
       0.81125053, 0.79803888, 0.7259333 , 0.6058237 , 0.63609493,
       0.68554449, 0.61041372, 0.54039646, 0.5675648 , 0.60125367,
       0.68312597, 0.74722062, 0.73914834, 0.75690716]
avg_hourly_acceptance = sum(hourly_acceptance_rates) / 24;

def get_next_goal(current_trips):
    reward_tier = max(-1, (current_trips - base_reward_threshold) // target_increment) + 1
    new_goal = base_reward_threshold + reward_tier * target_increment
    new_reward = base_reward + reward_increment * (reward_tier ** reward_boost_factor)

    return new_goal, new_reward
