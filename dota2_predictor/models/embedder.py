import torch.nn as nn
import torch


class HeroEmbeddings(nn.Module):
    def __init__(self,embedding_dim,projection_dim,output_size):
        super().__init__()
        self.primary_attr = nn.Embedding(4,embedding_dim)
        self.attack_type = nn.Embedding(2,embedding_dim)
        self.roles_embed = nn.Linear(8,embedding_dim)
        self.projection = nn.Sequential(nn.Linear(22,projection_dim),nn.ReLU(),nn.Linear(projection_dim,projection_dim//2))
        total_size = 3*embedding_dim+(projection_dim//2)
        self.combine = nn.Sequential(nn.Linear(total_size,output_size),nn.ReLU(),nn.LayerNorm(output_size))


    def forward(self,p_attrs,a_types,role_i,float_stats):
        pa_embed = self.primary_attr(p_attrs)
        attack_embed = self.attack_type(a_types)
        role_embed = self.roles_embed(role_i)
        stats_embed = self.projection(float_stats)
        hero = torch.cat([pa_embed,attack_embed,role_embed,stats_embed],dim=1)
        hero = self.combine(hero)
        return hero
    
class TeamComp(nn.Module):
    def __init__(self,embedding_dim,num_attention,output_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=num_attention,batch_first=True)
        self.output = nn.Linear(embedding_dim,output_dim)

    def forward(self,hero_picks):
        attn_weigh, _ = self.attention(hero_picks,hero_picks)
        team = attn_weigh.mean(dim=1)
        match = self.output(team)
        return match
    


