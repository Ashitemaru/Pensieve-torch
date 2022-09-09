import torch

from config import config
from model.network import ActorNet, CriticNet

class A3C:
    def __init__(
        self,
        is_central,
        n_feature,
        n_action,
        actor_lr = config["actor_default_lr"],
        critic_lr = config["critic_default_lr"],
        reward_decay = config["default_reward_decay"],
        init_entropy_weight = config["default_entropy_weight"],
        entropy_decay = config["entropy_decay"],
    ):
        self.n_feature = n_feature
        self.n_action = n_action
        self.gamma = reward_decay
        self.beta = init_entropy_weight
        self.beta_decay = entropy_decay
        self.is_central = is_central
        self.device = torch.device("cpu")

        self.actor = ActorNet(
            n_feature = self.n_feature,
            n_action = self.n_action
        ).to(self.device)
        if self.is_central:
            self.actor_optimizer = torch.optim.RMSprop(
                params = self.actor.parameters(),
                lr = actor_lr,
                alpha = 0.9,
                eps = 1e-10,
            )
            self.actor_optimizer.zero_grad()

            self.critic = CriticNet(
                n_feature = self.n_feature,
                n_action = self.n_action
            ).to(self.device)
            self.critic_optimizer = torch.optim.RMSprop(
                params = self.critic.parameters(),
                lr = critic_lr,
                alpha = 0.9,
                eps = 1e-10,
            )
            self.critic_optimizer.zero_grad()
        else:
            self.actor.eval()

        self.loss_function = torch.nn.MSELoss()

    def select_action(self, state):
        if not self.is_central:
            with torch.no_grad():
                action_prob = self.actor(state.to(self.device))
                prob_generator = torch.distributions.Categorical(action_prob)
                return prob_generator.sample().item()
        else:
            # TODO: What should be done here?
            pass

    def backward_gradients(self, state_batch, action_batch, reward_batch):
        """
        state_batch.size(): (batch_size, param_count, past_video_chunk_num)
        action_batch.size(): (batch_size, )
        reward_batch.size(): (batch_size, )

        param_count: 6.
        past_video_chunk_num: default to 8, denoted as 'k' according to the paper.
        """
        state_batch = torch.cat(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch).to(self.device)

        accumulate_reward_batch = torch.zeros(reward_batch.shape).to(self.device)
        accumulate_reward_batch[-1] = reward_batch[-1]
        for t in reversed(range(reward_batch.shape[0] - 1)):
            accumulate_reward_batch[t] = reward_batch[t] + self.gamma * accumulate_reward_batch[t + 1]

        # Calc TD error
        with torch.no_grad():
            value_batch = self.critic(state_batch.to(self.device)).to(self.device) # Shape: (batch_size, )
        td_error_batch = accumulate_reward_batch - value_batch # Shape: (batch_size, )

        action_prob = self.actor(state_batch.to(self.device)) # Shape: (batch_size, n_action)
        prob_generator = torch.distributions.Categorical(action_prob)
        log_action_prob = prob_generator.log_prob(action_batch) # Shape: (batch_size, )
        actor_loss_base = torch.sum(log_action_prob * (-td_error_batch)) # Shape: scalar

        entropy_regularization = -self.beta * torch.sum(prob_generator.entropy()) # Shape: scalar
        actor_loss = actor_loss_base + entropy_regularization
        actor_loss.backward()

        # TODO: Is this necessary to re-forward?
        critic_loss = self.loss_function(
            accumulate_reward_batch,
            self.critic(state_batch.to(self.device))
        ) # Shape: scalar
        critic_loss.backward()

    def hard_update_actor(self, new_param):
        for tgt_param, src_param in zip(self.actor.parameters(), new_param):
            tgt_param.data.copy_(src_param.data)

    def update_net(self):
        # Accumulate all the gradients and update once
        if self.is_central:
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

    def decay_entropy(self):
        self.beta *= self.beta_decay

if __name__ == "__main__":
    pass
