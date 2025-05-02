# voting_agent.py
import random
import torch

class VotingAgent:
    def __init__(self, dqn_agent, ppo_agent):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn_agent = dqn_agent
        self.ppo_agent = ppo_agent

        self.dqn_agent.model.to(self.device).eval()
        self.ppo_agent.model.to(self.device).eval()

    def rule_based_vote(self, action_dqn, action_ppo, log_prob):
        """Confidence 기반 Voting 룰 (Soft Voting)"""
        confidence = log_prob.exp().item()
        if confidence > 0.6:
            return action_ppo
        else:
            return action_dqn

    def act(self, state):
        action_dqn = self.dqn.act(state)
        action_ppo, log_prob, _ = self.ppo.get_action(state)

        if action_dqn == action_ppo:
            return action_dqn
        else:
            return self.rule_based_vote(action_dqn, action_ppo, log_prob)
