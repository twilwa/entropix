from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCache:
    k: jax.Array
    v: jax.Array

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        return cls(
            k=jnp.zeros((layers, bsz, kv_heads, max_seq_len, head_dim), dtype=jnp.bfloat16),
            v=jnp.zeros((layers, bsz, kv_heads, max_seq_len, head_dim), dtype=jnp.bfloat16)
        )

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int):
        # Update the caches at the current position
        ck = self.k.at[layer_idx, :, :, cur_pos, :].set(xk)
        cv = self.v.at[layer_idx, :, :, cur_pos, :].set(xv)
        
        # Extract the keys and values up to the current position
        keys = ck[layer_idx, :, :, :cur_pos + xk.shape[1], :]
        values = cv[layer_idx, :, :, :cur_pos + xv.shape[1], :]
        
        return keys, values, KVCache(k=ck, v=cv)

