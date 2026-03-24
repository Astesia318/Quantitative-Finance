import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from .feature_extractor import FeatureExtractor
from .factor_encoder import FactorEncoder
from .factor_decoder import FactorDecoder
from .factor_predictor import FactorPredictor


# factor_VAE.py 的修改示例
class FactorVAE(nn.Module):
    def __init__(self, num_feature, num_factor, num_portfolio, hidden_size,seq_len):
        super(FactorVAE, self).__init__()

        # Feature Extractor 只需要知道输入特征数和输出隐变量数
        self.feature_extractor = FeatureExtractor(
            num_input=num_feature,    # 原 gru_input_size
            hidden_size=hidden_size,    
            seq_len=seq_len
        )
        
        # Encoder 需要隐变量数、因子数、以及构建组合的 M
        self.factor_encoder = FactorEncoder(
            hidden_size=hidden_size,
            num_factor=num_factor, 
            num_portfolio=num_portfolio
        )
        
        # Predictor 只需要隐变量数和因子数
        self.factor_predictor = FactorPredictor(
            num_factor=num_factor,
            hidden_size=hidden_size
        )
        
        # Decoder 将因子转化为重建收益率
        self.factor_decoder = FactorDecoder(
            num_factor=num_factor,
            hidden_size=hidden_size
        )

    def run_model(self, characteristics, future_returns, gamma=1,mask=None):
        latent_features = self.feature_extractor(characteristics)
        # (batch_size, stock_size, latent_size)
        eps=1e-4
        mu_post, sigma_post = self.factor_encoder(latent_features, future_returns)
        # (batch_size, factor_size)
        sigma_post = torch.clamp(sigma_post,min=eps)
        m_encoder = Normal(mu_post, sigma_post)
        factors_post = m_encoder.sample()

        # (batch_size, factor_size, 1)

        reconstruct_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
            factors_post, latent_features
        )

        mu_dec, sigma_dec = self.get_decoder_distribution(
            mu_alpha, sigma_alpha, mu_post, sigma_post, beta
        )
        
        sigma_dec = torch.clamp(sigma_dec,min=eps)
       
 
        log_prob_matrix = Normal(mu_dec, sigma_dec).log_prob(future_returns.unsqueeze(-1))
        if mask is not None:
            log_prob_matrix = log_prob_matrix * mask.unsqueeze(-1)
        
        loss_negloglike = -log_prob_matrix.mean()
        # valid_count = mask.sum() if mask is not None else (self.stock_size * latent_features.shape[0])

        # latent_features.shape[0] is the batch_size

        mu_prior, sigma_prior = self.factor_predictor(latent_features)
        sigma_prior = torch.clamp(sigma_prior,min=eps)
        m_predictor = Normal(mu_prior, sigma_prior)
        
        loss_KL = kl_divergence(m_encoder, m_predictor).mean()
        # print(f"loss_neg:{loss_negloglike},loss_KL:{loss_KL}")
        loss = loss_negloglike + gamma * loss_KL

        return loss

    def prediction(self, characteristics):
        with torch.no_grad():

            latent_features = self.feature_extractor(characteristics)

            mu_prior, sigma_prior = self.factor_predictor(latent_features)

            m_prior = Normal(mu_prior, sigma_prior)
            factor_prior = m_prior.sample()

            pred_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
                factor_prior, latent_features
            )

            mu_dec, sigma_dec = self.get_decoder_distribution(
                mu_alpha, sigma_alpha, mu_prior, sigma_prior, beta
            )

        return pred_returns, mu_dec, sigma_dec

    def get_decoder_distribution(
        self, mu_alpha, sigma_alpha, mu_factor, sigma_factor, beta
    ):
        # print(mu_alpha.shape, mu_factor.shape, sigma_factor.shape, beta.shape)
        mu_dec = mu_alpha + torch.bmm(beta, mu_factor)

        sigma_dec = torch.sqrt(
            torch.square(sigma_alpha)
            + torch.bmm(torch.square(beta), torch.square(sigma_factor))
        )

        return mu_dec, sigma_dec
