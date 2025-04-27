import torch.nn as nn
import torch


class HeroEmbeddings(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.primary_attr = nn.Embedding(4,embedding_dim)
        self.attack_type = nn.Embedding(2,embedding_dim)
        self.roles_embed = nn.Linear(8,embedding_dim)
        self.projection = nn.Linear(22,22)

    def forward(self,p_attrs,a_types,role_i,float_stats):
        pa_embed = self.primary_attr(p_attrs)
        attack_embed = self.attack_type(a_types)
        role_embed = self.roles_embed(role_i)
        stats_embed = self.projection(float_stats)

        
        batch_size = pa_embed.shape[0] // 10
        device = pa_embed.device
        team_flag = torch.cat([torch.ones(batch_size*5,1,device=device),torch.zeros(batch_size*5,1,device=device)],dim=0)
        hero = torch.cat([pa_embed,attack_embed,role_embed,stats_embed,team_flag],dim=1)
        return hero.squeeze(0)
