#Importing library 
import torch
import torch.optim as optim
import torch.nn.functional as F

from data_preprocessing import data_preprocess
from env import TradingEnvironment
from policy import Policy


gamma = 0.9
log_interval = 60

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + (gamma * R)
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    
    epsilon = (torch.rand(1) / 1e4) - 5e-5
    # With different architectures, I found the following standardization step sometimes
    # helpful, sometimes unhelpful.
    # rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)
    # Alternatively, comment it out and use the following line instead:
    rewards += epsilon
    
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = torch.tensor(r - value.item())
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
        
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss = torch.clamp(loss, -1e-5, 1e5)
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

if __name__ == '__main__':

    apl_stock, apl_open, apl_close, msf_stock, msf_open, msf_close = data_preprocess()
    print('Data collecting and preprocessing complete')
    env = TradingEnvironment(apl_open, apl_close, msf_open, msf_close, max_stride=4, series_length=250, starting_cash_mean=1000, randomize_cash_std=100, starting_shares_mean=100, randomize_shares_std=10)
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    env.reset()
    
    running_reward = 0
    for episode in range(0, 4000):
        state = env.reset()
        reward = 0
        done = False
        msg = None
        while not done:
            action = model.act(state, env)
            state, reward, done, msg = env.step(action)
            model.rewards.append(reward)
            if done:
                break
        running_reward = running_reward * (1 - 1/log_interval) + reward * (1/log_interval)
        finish_episode()
        # Resetting the hidden state seems unnecessary - it's effectively random from the previous
        # episode anyway, more random than a bunch of zeros.
        # model.reset_hidden()
        if msg["msg"] == "done" and env.portfolio_value() > env.starting_portfolio_value * 1.1 and running_reward > 500:
            print("Early Stopping: " + str(int(reward)))
            break
        if episode % log_interval == 0:
            print("""Episode {}: started at {:.1f}, finished at {:.1f} because {} @ t={}, \
    last reward {:.1f}, running reward {:.1f}""".format(episode, env.starting_portfolio_value, \
                env.portfolio_value(), msg["msg"], env.cur_timestep, reward, running_reward))
    
    apl_open_orig = apl_stock["Open"].values
    apl_close_orig = apl_stock["Close"].values
    msf_open_orig = msf_stock["Open"].values
    msf_close_orig = msf_stock["Close"].values
    apl_open_orig[:108] /= 7
    apl_close_orig[:108] /= 7
    
    complete_game = False
    while not complete_game:
        bought_apl_at = []
        bought_msf_at = []
        sold_apl_at = []
        sold_msf_at = []
        bought_apl_at_orig = []
        bought_msf_at_orig = []
        sold_apl_at_orig = []
        sold_msf_at_orig = []
        nothing_at = []
        ba_action_times = []
        bm_action_times = []
        sa_action_times = []
        sm_action_times = []
        n_action_times = []
        starting_val = env.starting_portfolio_value
        print("Starting portfolio value: {}".format(starting_val))
        for i in range(0,env.series_length + 1):
            action = model.act(env.state, env)
            if action == 0:
                bought_apl_at.append(apl_open[env.cur_timestep])
                bought_apl_at_orig.append(apl_open_orig[env.cur_timestep])
                ba_action_times.append(env.cur_timestep)
            if action == 1:
                sold_apl_at.append(apl_close[env.cur_timestep])
                sold_apl_at_orig.append(apl_close_orig[env.cur_timestep])
                sa_action_times.append(env.cur_timestep)
            if action == 2:
                nothing_at.append(35)
                n_action_times.append(env.cur_timestep)
            if action == 3:
                bought_msf_at.append(msf_open[env.cur_timestep])
                bought_msf_at_orig.append(msf_open_orig[env.cur_timestep])
                bm_action_times.append(env.cur_timestep)
            if action == 4:
                sold_msf_at.append(msf_close[env.cur_timestep])
                sold_msf_at_orig.append(msf_close_orig[env.cur_timestep])
                sm_action_times.append(env.cur_timestep)
            next_state, reward, done, msg = env.step(action)
            if msg["msg"] == 'bankrupted self':
                env.reset()
                break
            if msg["msg"] == 'sold more than have':
                env.reset()
                break
            if msg["msg"] == "done":
                print("{}, have {} aapl and {} msft and {} cash".format(msg["msg"], next_state[0], next_state[1], next_state[2]))
                val = env.portfolio_value()
                print("Finished portfolio value {}".format(val))
                if val > starting_val * 1.1: complete_game = True
                env.reset()
                break 