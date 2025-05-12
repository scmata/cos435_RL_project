import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



DMC_env_names=[
    'acrobot/swingup', 
    'ball_in_cup/catch', 'cartpole/balance', 'cartpole/balance_sparse',
    'cartpole/swingup', 'cartpole/swingup_sparse', 'cheetah/run', 'dog_run', 'dog_stand', 'dog_trot',
]

DMC_env_names2=[
    'ball_in_cup/catch', 'cartpole/balance', 'cartpole/balance_sparse',
    'cartpole/swingup', 'cheetah/run', 'dog_run', 'dog_stand', 'dog_trot',
]

#MRQ vs TD3 for acrobot/swingup: 212.14x
#MRQ reward: 113.27, TD3 reward: 0.53

#MRQ vs TD3 for cartpole/swingup_sparse: infx
#MRQ reward: 286.80, TD3 reward: 0.00

GYM_TASKS = [
    'Ant-v4',
    'HalfCheetah-v4',
    'Humanoid-v4',
]

#maybe do bipedal walker and mountain car cotinuous instead for gym


#safe_env_name = env_name.replace("/", "_")

#four groups of files
#MR Q -DMC e.g. results/MRQ/DMC/acrobot_swingup_evaluations_MRQ_A.csv
#MR Q-Gym  e.g. results/MRQ/Gym/Ant-v4_evaluations_MRQ_B.csv
#TD3 -DMC e.g. results/TD3/DMC/acrobot_swingup_evaluations_TD3_A.csv
#TD3 -Gym e.g. results/TD3/Gym/Ant-v4_evaluations_TD3_B.csv

def compare_rewards(env_names, file_path_template, suffix):
    results = []
    for env_name in env_names:
        safe_env_name = env_name.replace("/", "_")
        # Load the data
        df_MRQ = pd.read_csv(file_path_template.format(algorithm="MRQ", env=safe_env_name, suffix=suffix))
        df_TD3 = pd.read_csv(file_path_template.format(algorithm="TD3", env=safe_env_name, suffix=suffix))

        mr_q_reward = df_MRQ['Reward'][-10:].mean()
        td3_reward = df_TD3['Reward'][-10:].mean()

        normalised_reward = mr_q_reward / td3_reward
        print(f"MRQ vs TD3 for {env_name}: {normalised_reward:.2f}x")

        print(f"MRQ reward: {mr_q_reward:.2f}, TD3 reward: {td3_reward:.2f}")

        results.append(normalised_reward)
    return results


dmc_results = compare_rewards(
    DMC_env_names2,
    file_path_template="results/{algorithm}/DMC/{env}_evaluations_{algorithm}_A.csv",
    suffix="A"
)

gym_results = compare_rewards(
    GYM_TASKS,
    file_path_template="results/{algorithm}/Gym/{env}_evaluations_{algorithm}_A.csv",
    suffix="A"
)




# Create a single figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 6))

# Subplot 1: Normalized DMC Results
# Bar plot for normalized rewards
axes[0].bar(DMC_env_names2, dmc_results, color='blue')
axes[0].axhline(y=1, color='red', linestyle='--', linewidth=1)  # Horizontal line at y=1
axes[0].set_title('Normalized DMC Results')
axes[0].set_ylabel('Normalized Reward')
axes[0].set_xlabel('Environment')
axes[0].tick_params(axis='x', labelrotation=70)

# Subplot 2: Absolute DMC Rewards
# Calculate absolute rewards for MRQ and TD3
mrq_rewards = [pd.read_csv(f"results/MRQ/DMC/{env.replace('/', '_')}_evaluations_MRQ_A.csv")['Reward'][-10:].mean() for env in DMC_env_names]
td3_rewards = [pd.read_csv(f"results/TD3/DMC/{env.replace('/', '_')}_evaluations_TD3_A.csv")['Reward'][-10:].mean() for env in DMC_env_names]

# Bar plot for absolute rewards
x = np.arange(len(DMC_env_names))
bar_width = 0.35
axes[1].bar(x - bar_width / 2, mrq_rewards, bar_width, label='MRQ', color='blue')
axes[1].bar(x + bar_width / 2, td3_rewards, bar_width, label='TD3', color='orange')

axes[1].set_title('Absolute DMC Rewards')
axes[1].set_ylabel('Reward')
axes[1].set_xlabel('Environment')
axes[1].set_xticks(x)
axes[1].set_xticklabels(DMC_env_names, rotation=70)
axes[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# Save the combined plot to the 'plots' directory
plt.savefig("plots/overall/dmc_combined_results.png")
print(f"Combined DMC plot saved")



# Create a grid of subplots for reward over episode number for MRQ on DMC environments
fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # 3x3 grid
axes = axes.flatten()  # Flatten to easily iterate over

for i, env_name in enumerate(DMC_env_names):
    if i >= len(axes):  # Ensure we don't exceed the number of subplots
        break
    safe_env_name = env_name.replace("/", "_")
    # Load the data
    df_MRQ = pd.read_csv(f"results/MRQ/DMC/{safe_env_name}_evaluations_MRQ_A.csv")
    df_TD3 = pd.read_csv(f"results/TD3/DMC/{safe_env_name}_evaluations_TD3_A.csv")
    
    # Plot reward over episode number for MRQ
    axes[i].plot(df_MRQ['Episode'], df_MRQ['Reward'], label='MRQ', color='blue')
    # Plot reward over episode number for TD3
    axes[i].plot(df_TD3['Episode'], df_TD3['Reward'], label='TD3', color='orange')
    
    axes[i].set_title(f"Reward for {env_name}")
    axes[i].set_xlabel("Episode")
    axes[i].set_ylabel("Reward")
    axes[i].grid(True)
    axes[i].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# Save the grid plot to the 'plots' directory
plt.savefig("plots/overall/mrq_dmc_rewards.png")
print(f"Grid plot saved")


# Create a grid of subplots for reward over episode number for MRQ and TD3 on Gym environments
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid
axes = axes.flatten()  # Flatten to easily iterate over

for i, env_name in enumerate(GYM_TASKS):
    if i >= len(axes):  # Ensure we don't exceed the number of subplots
        break
    safe_env_name = env_name.replace("/", "_")
    # Load the data
    df_MRQ = pd.read_csv(f"results/MRQ/Gym/{safe_env_name}_evaluations_MRQ_A.csv")
    df_TD3 = pd.read_csv(f"results/TD3/Gym/{safe_env_name}_evaluations_TD3_A.csv")
    
    # Plot reward over episode number for MRQ
    axes[i].plot(df_MRQ['Episode'], df_MRQ['Reward'], label='MRQ', color='blue')
    # Plot reward over episode number for TD3
    axes[i].plot(df_TD3['Episode'], df_TD3['Reward'], label='TD3', color='orange')
    
    axes[i].set_title(f"Reward for {env_name}")
    axes[i].set_xlabel("Episode")
    axes[i].set_ylabel("Reward")
    axes[i].grid(True)
    axes[i].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# Save the grid plot to the 'plots' directory
plt.savefig("plots/overall/mrq_td3_gym_rewards.png")
print(f"Grid plot2 saved")



# Create bar plots for DMC and Gym results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Bar plot for DMC results
axes[0].bar(DMC_env_names, dmc_results, color='blue')
axes[0].axhline(y=1, color='red', linestyle='--', linewidth=1)  # Horizontal line at y=1
axes[0].set_title('DMC Results')
axes[0].set_ylabel('Normalized Reward')
axes[0].set_xlabel('Environment')
axes[0].tick_params(axis='x', labelrotation=45)

# Bar plot for Gym results
axes[1].bar(GYM_TASKS, gym_results, color='green')
axes[1].axhline(y=1, color='red', linestyle='--', linewidth=1)  # Horizontal line at y=1
axes[1].set_title('Gym Results')
axes[1].set_ylabel('Normalized Reward')
axes[1].set_xlabel('Environment')
axes[1].tick_params(axis='x', labelrotation=45)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# Save the plot to the 'plots' directory
plt.savefig("plots/overall/comparison_results.png")
print(f"Plot saved")
