import gym
import gym_holdem

import sys

from gym_holdem.envs.holdem_env import HoldemEnv

from poker_ai.agents import DQNAgent

epochs = 1000
trial_len = 500


def main():
    model = ""
    target_model = ""
    print(sys.argv)
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    if len(sys.argv) >= 3:
        target_model = sys.argv[2]

    env = gym.make("poker_ai_gym-v0")
    gamma = 0.9
    epsilon = .95

    # updateTargetNetwork = 1000
    dqn_agents = [DQNAgent(env=env, player=env.players[idx], model_path=model, target_model_path=target_model) for idx in range(env.player_amount)]
    for epoch in range(epochs):
            print(f"### Epoch {epoch} ###")
            for idx, agent in enumerate(dqn_agents):
                print(f"Agent {idx}: win_count=={agent.win_count}")
            play_game(dqn_agents, env)
    
    win_counts = [a.win_count for a in dqn_agents]
    highest_win_count = max(win_counts)
    best_agent_idx = win_counts.index(highest_win_count)
    best_agent = dqn_agents[best_agent_idx]
    best_agent.save_model("trained_models/best_model.h5")
    best_agent.save_target_model("trained_models/best_target_model.h5")


def next_agent(dqn_agents, env):
    for a in dqn_agents:
        if a.player == env.table.next_player:
            return a
    return None


def play_game(dqn_agents, env):
    cur_state = env.reset()
    for step in range(trial_len):
        dqn_agent = next_agent(dqn_agents, env)

        action = dqn_agent.act(cur_state)

        new_state, reward, done, debug = env.step(action)
        env.render()

        # reward = reward if not done else -20
        # new_state = new_state.reshape(1,2)
        dqn_agent.remember(cur_state, action, reward, new_state, done)
            
        dqn_agent.replay()       # internally iterates default (prediction) model
        dqn_agent.target_train() # iterates target model

        cur_state = new_state
        if done:
            winner = env.table_players[0]
            for a in dqn_agents:
                if a.player == winner:
                    a.win_count += 1
                    break
            return


if __name__ == "__main__":
    main()
