import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class base_competition(nn.Module):
    def __init__(self, agents, c, r, datastream):
        super(base_competition, self).__init__()
        self.agents = agents
        self.datastream = datastream
        self.c = c
        self.r = r

    def run(self, data, logger):
        pass

class classification_competition(base_competition):

    def __init__(self, agents, datastream, c=1, r=2):
        super().__init__(agents, c, r, datastream)

    def _process_data(self, data):
        x,y = data
        #x = x.reshape(-1,1)
        y = y.float()
        return x,y 

    def score_models(self, data):
        x,y = self._process_data(data)

        self.logger['x'].append(x)
        self.logger['y'].append(y)

        y_hats = []
        #compute predicted vals
        scores = []
        for i,a in enumerate(self.agents):
            a.get_reward(-self.c)
            y_hat = a.predict(x)
            y_hats.append(y_hat)
            scores.append(self.score(y,y_hat))
        self.logger['scores'].append(scores)
        return scores, y_hats, x, y

    def score_test_models(self, test_data):
        x,y = self._process_data(test_data)
        #compute predicted vals
        scores = []

        for i,agent in enumerate(self.agents):
            agent.get_reward(-self.c)
            y_hat = agent.predict(x)
            scores.append(self.score(y,y_hat))
            self.logger['agents'][agent.id]['test_pred'].append(y_hat)
        self.logger['test_y'].append(y)
        return scores

    def system_correctness(self, scores):
        correct_agents = []
        for i,score in enumerate(scores):
            if score == 1:
                correct_agents.append(self.agents[i])

        if correct_agents == []:
            self.logger['agg-correct'].append(False)
            correct_agents = self.agents
        else:
            self.logger['agg-correct'].append(True)   

        return correct_agents

    def user_decision(self, scores):
        correct_agents = self.system_correctness(scores)
        wid = torch.randint(len(correct_agents), (1,))[0]
        return correct_agents[wid]

    def update_winner(self, winner, x, y):
        self.logger['winner'].append(winner.id)
        winner.get_reward(self.r)
        winner.add_data(x,y)
        winner._update(x,y)
        winner.wins += 1

    def update_agents(self, y_hats):
        for i,agent in enumerate(self.agents):
            self.logger['agents'][agent.id]['reward'].append(agent.reward)
            self.logger['agents'][agent.id]['wins'].append(agent.wins)
            self.logger['agents'][agent.id]['y_hat'].append(y_hats[i])
            self.logger['agents'][agent.id]['dataset_counts'] = agent.dataset_counts
    
    def run(self, data, logger):
        scores, y_hats, x, y = self.score_models(data)
        winner = self.user_decision(scores)
        self.update_winner(winner, x, y)
        self.update_agents(y_hats)

    def score(self, y, y_hat):
        return  1 if y == y_hat else 0

    def _last_train(self):
        print("LAST TRAINING", flush=True)
        for i,agent in enumerate(self.agents):
            agent._seed_train()

class data_purchase_classification_competition(classification_competition):
   
    def __init__(self, agents, datastream, c=0, r=1, alpha=1):
        super().__init__(agents, datastream, c=c, r=r)
        self.alpha = alpha

    def system_correctness(self, scores, wid):
        corr_agent = super().system_correctness(scores)
        self.logger['agg-correct'][-1] = self.logger['agg-correct'][-1] and \
                                        wid in [a.id for a in corr_agent]

    def predict_and_purchase(self, data):
        x,y = self._process_data(data)

        self.logger['x'].append(x)
        self.logger['y'].append(y)

        y_hats, purchases = [], []
        #compute predicted vals
        scores = []
        for i,a in enumerate(self.agents):
            # prediction
            y_hat_prob = a.predict(x, prob=True)
            _, y_hat = torch.max(y_hat_prob, 1)
            y_hats.append(y_hat) # class prediction

            # buying based on a model prediction
            purchase = a.purchase(y_hat_prob)
            purchases.append(purchase)

            # quality score is based on predictions
            scores.append(self.score(y,y_hat)) 
        self.logger['scores'].append(scores)
        return scores, purchases, y_hats, x, y

    def user_decision(self, scores, purchases):
        if sum(purchases) >= 1:
            candidates = np.where(np.array(purchases)==1)[0]
            wid = np.random.choice(candidates, size=1)[0]
        else:
            s_np = np.array(scores)
            softmin = np.exp(self.alpha*s_np)/sum(np.exp(self.alpha*s_np))
            wid = np.random.choice(range(len(s_np)), size=1, p=softmin)[0]
        self.system_correctness(scores, wid)
        return self.agents[wid]

    def update_winner(self, winner, x, y, purchases):
        super().update_winner(winner, x, y)
        if sum(purchases) >= 1:
            winner.currnet_budget -= 1 # budget system
            self.logger['buyer'].append(winner.id)
        else:
            self.logger['buyer'].append(-1)

    def run(self, data, logger):
        scores, purchases, y_hats, x, y = self.predict_and_purchase(data)
        winner = self.user_decision(scores, purchases)
        self.update_winner(winner, x, y, purchases)
        self.update_agents(y_hats)

