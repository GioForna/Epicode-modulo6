import torch

from sandbox.config import TrainConfig
from sandbox.trainer import SyntheticDataModel
from sandbox.vae import TabularVAE, vae_loss


def test_forward_shapes():
    model = TabularVAE(input_dim=10, latent_dim=4, hidden_dims=(16, 8))
    x = torch.rand(32, 10)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (32, 4)
    assert recon.min() >= 0 and recon.max() <= 1


def test_loss_is_mse_plus_kl():
    x = torch.rand(8, 5)
    mu = torch.zeros(8, 3)
    logvar = torch.zeros(8, 3)
    total, recon, kl = vae_loss(x, x, mu, logvar)
    # ricostruzione perfetta e posterior uguale alla prior: loss nulla
    assert recon.item() == 0 and abs(kl.item()) < 1e-7 and total.item() == 0


def test_reparameterization_has_gradient():
    mu = torch.zeros(4, 2, requires_grad=True)
    logvar = torch.zeros(4, 2, requires_grad=True)
    z = TabularVAE.reparameterize(mu, logvar)
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None


def test_train_generate_save_load(tmp_path, mixed_df):
    model = SyntheticDataModel(TrainConfig(epochs=5, batch_size=32)).fit(mixed_df)
    assert len(model.history) == 5
    synth = model.generate(50, seed=1)
    assert synth.shape == (50, mixed_df.shape[1])
    assert list(synth.columns) == list(mixed_df.columns)

    model.save(tmp_path)
    loaded = SyntheticDataModel.load(tmp_path)
    again = loaded.generate(50, seed=1)
    assert again.equals(synth)
