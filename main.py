# mutator.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# -----------------------
# Mutator definition
# -----------------------
class Mutator(nn.Module):
    genome_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, seq):
        """
        seq: [batch, seq_len, feature_dim] input sequence
        genome: trainable vector
        returns: mutated sequence [batch, seq_len, output_dim]
        """
        # Genome as a learnable parameter
        genome = self.param("genome", nn.initializers.normal(), (self.genome_dim,))

        # Simple mapping: concatenate genome to input
        genome_broadcast = jnp.broadcast_to(genome, (seq.shape[0], seq.shape[1], self.genome_dim))
        x = jnp.concatenate([seq, genome_broadcast], axis=-1)

        # Small network head to produce mutated sequence
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# -----------------------
# Training utilities
# -----------------------
class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, genome_dim, output_dim, lr=1e-3):
    model = Mutator(genome_dim=genome_dim, output_dim=output_dim)
    params = model.init(rng, jnp.ones((1, 5, 8)))  # dummy input: batch=1, seq_len=5, feature_dim=8
    tx = optax.adam(lr)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# -----------------------
# Example training step
# -----------------------
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        preds = state.apply_fn(params, batch)
        # Dummy loss: encourage small outputs
        return jnp.mean(preds**2)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, genome_dim=16, output_dim=8)

    # Dummy input batch: batch=32, seq_len=5, feature_dim=8
    batch = jax.random.normal(rng, (32, 5, 8))

    for step in range(10):
        state, loss = train_step(state, batch)
        if step % 2 == 0:
            print(f"Step {step}, loss={loss:.4f}")
